-- ============================================================
-- Fraud Detection — SQL Threshold Tracking Queries
-- Author: Nithin Venkatesh
-- Description: KPI monitoring queries replicating production-
--              style fraud threshold tracking across seller/
--              transaction accounts.
-- ============================================================


-- ─── 1. DAILY FRAUD KPI SUMMARY ─────────────────────────────────────────────
-- Core dashboard query: fraud rate, recovery rate, case turnaround by day.
-- Equivalent to the threshold tracking queries maintained at Amazon.

SELECT
    DATE(transaction_time)                                          AS report_date,
    COUNT(*)                                                        AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END)                  AS fraud_cases,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) * 100.0
        / COUNT(*)                                                  AS fraud_rate_pct,
    SUM(CASE WHEN is_fraud = 1 AND is_recovered = 1 THEN 1 ELSE 0 END) * 100.0
        / NULLIF(SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END), 0) AS recovery_rate_pct,
    AVG(CASE WHEN is_fraud = 1
        THEN DATEDIFF(resolved_time, flagged_time) END)             AS avg_case_turnaround_days
FROM transactions
WHERE transaction_time >= DATEADD(day, -30, GETDATE())
GROUP BY DATE(transaction_time)
ORDER BY report_date DESC;


-- ─── 2. THRESHOLD DRIFT DETECTION ───────────────────────────────────────────
-- Compares current week fraud rate to rolling 4-week average.
-- Flags accounts where rate has drifted beyond acceptable band (±1.5%).
-- This is the query pattern used to catch threshold drift before SLA breach.

WITH weekly_rates AS (
    SELECT
        account_id,
        DATE_TRUNC('week', transaction_time)                        AS week_start,
        COUNT(*)                                                    AS total_txns,
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) * 100.0
            / COUNT(*)                                              AS fraud_rate_pct
    FROM transactions
    WHERE transaction_time >= DATEADD(week, -5, GETDATE())
    GROUP BY account_id, DATE_TRUNC('week', transaction_time)
),
rolling_avg AS (
    SELECT
        account_id,
        week_start,
        fraud_rate_pct,
        AVG(fraud_rate_pct) OVER (
            PARTITION BY account_id
            ORDER BY week_start
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
        )                                                           AS rolling_4wk_avg
    FROM weekly_rates
)
SELECT
    account_id,
    week_start,
    ROUND(fraud_rate_pct, 2)                                        AS current_fraud_rate,
    ROUND(rolling_4wk_avg, 2)                                       AS rolling_avg_fraud_rate,
    ROUND(fraud_rate_pct - rolling_4wk_avg, 2)                      AS drift,
    CASE
        WHEN ABS(fraud_rate_pct - rolling_4wk_avg) > 1.5 THEN 'ALERT — Threshold Drift'
        WHEN ABS(fraud_rate_pct - rolling_4wk_avg) > 0.8 THEN 'WATCH'
        ELSE 'Normal'
    END                                                             AS drift_status
FROM rolling_avg
WHERE week_start = DATE_TRUNC('week', GETDATE())
ORDER BY ABS(fraud_rate_pct - COALESCE(rolling_4wk_avg, fraud_rate_pct)) DESC;


-- ─── 3. FALSE POSITIVE ANALYSIS ─────────────────────────────────────────────
-- Segments false positive alerts by transaction type and amount range.
-- Used for root cause analysis when tuning detection thresholds.

SELECT
    transaction_type,
    CASE
        WHEN amount < 50    THEN 'Under $50'
        WHEN amount < 200   THEN '$50–$200'
        WHEN amount < 1000  THEN '$200–$1K'
        ELSE 'Over $1K'
    END                                                             AS amount_bucket,
    COUNT(*)                                                        AS false_positive_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()                       AS pct_of_total_fp,
    AVG(fraud_score)                                                AS avg_fraud_score,
    AVG(amount)                                                     AS avg_txn_amount
FROM fraud_alerts
WHERE alert_outcome = 'FALSE_POSITIVE'
  AND alert_date >= DATEADD(month, -1, GETDATE())
GROUP BY transaction_type, amount_bucket
ORDER BY false_positive_count DESC;


-- ─── 4. SLA COMPLIANCE TRACKING ─────────────────────────────────────────────
-- Monitors case resolution times against SLA targets by queue priority.
-- Replicates the SLA compliance queries used during peak sale events.

WITH case_resolution AS (
    SELECT
        case_id,
        priority_tier,
        opened_at,
        resolved_at,
        DATEDIFF(hour, opened_at, resolved_at)                      AS resolution_hours,
        CASE priority_tier
            WHEN 'P1' THEN 4
            WHEN 'P2' THEN 24
            WHEN 'P3' THEN 72
        END                                                         AS sla_target_hours
    FROM fraud_cases
    WHERE opened_at >= DATEADD(month, -1, GETDATE())
      AND resolved_at IS NOT NULL
)
SELECT
    priority_tier,
    COUNT(*)                                                        AS total_cases,
    SUM(CASE WHEN resolution_hours <= sla_target_hours THEN 1 ELSE 0 END)
                                                                    AS within_sla,
    SUM(CASE WHEN resolution_hours > sla_target_hours THEN 1 ELSE 0 END)
                                                                    AS breached_sla,
    ROUND(SUM(CASE WHEN resolution_hours <= sla_target_hours THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2)                                      AS sla_compliance_pct,
    ROUND(AVG(resolution_hours), 1)                                 AS avg_resolution_hours,
    MAX(sla_target_hours)                                           AS sla_target_hours
FROM case_resolution
GROUP BY priority_tier
ORDER BY priority_tier;


-- ─── 5. CHARGEBACK & RECOVERY ANOMALY DETECTION ─────────────────────────────
-- Surfaces accounts with abnormal chargeback ratios vs. transaction volume.
-- Feeds into the Tableau dashboard for weekly anomaly review.

WITH account_summary AS (
    SELECT
        t.account_id,
        COUNT(t.transaction_id)                                     AS total_transactions,
        SUM(t.amount)                                               AS total_volume,
        COUNT(c.chargeback_id)                                      AS total_chargebacks,
        SUM(c.chargeback_amount)                                    AS total_chargeback_amount,
        SUM(CASE WHEN c.is_recovered = 1
            THEN c.chargeback_amount ELSE 0 END)                    AS recovered_amount
    FROM transactions t
    LEFT JOIN chargebacks c ON t.transaction_id = c.transaction_id
    WHERE t.transaction_time >= DATEADD(month, -3, GETDATE())
    GROUP BY t.account_id
),
anomaly_flags AS (
    SELECT
        *,
        total_chargebacks * 100.0
            / NULLIF(total_transactions, 0)                         AS chargeback_rate_pct,
        recovered_amount * 100.0
            / NULLIF(total_chargeback_amount, 0)                    AS recovery_rate_pct,
        AVG(total_chargebacks * 100.0 / NULLIF(total_transactions, 0))
            OVER ()                                                 AS avg_chargeback_rate
    FROM account_summary
)
SELECT
    account_id,
    total_transactions,
    ROUND(total_volume, 2)                                          AS total_volume,
    total_chargebacks,
    ROUND(chargeback_rate_pct, 2)                                   AS chargeback_rate_pct,
    ROUND(recovery_rate_pct, 2)                                     AS recovery_rate_pct,
    CASE
        WHEN chargeback_rate_pct > avg_chargeback_rate * 2 THEN 'HIGH ANOMALY'
        WHEN chargeback_rate_pct > avg_chargeback_rate * 1.5 THEN 'MODERATE'
        ELSE 'Normal'
    END                                                             AS anomaly_flag
FROM anomaly_flags
WHERE chargeback_rate_pct > 0
ORDER BY chargeback_rate_pct DESC;
