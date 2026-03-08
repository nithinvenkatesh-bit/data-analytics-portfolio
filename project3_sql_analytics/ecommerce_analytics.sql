-- ============================================================
-- E-Commerce SQL Analytics — Conversion Funnel, Cohort Analysis
-- & A/B Test Framework
-- Author: Nithin Venkatesh
-- Dataset: Brazilian E-Commerce (Olist) — Kaggle
-- https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
-- ============================================================


-- ═══════════════════════════════════════════════════════════
-- SECTION 1: CONVERSION FUNNEL ANALYSIS
-- Maps the full customer journey from session to purchase.
-- Pinpoints drop-off points by acquisition channel.
-- ═══════════════════════════════════════════════════════════

-- 1a. Full funnel — volume and conversion rate at each stage
WITH funnel AS (
    SELECT
        acquisition_channel,
        COUNT(DISTINCT session_id)                                   AS sessions,
        COUNT(DISTINCT CASE WHEN viewed_product = 1
            THEN session_id END)                                     AS product_views,
        COUNT(DISTINCT CASE WHEN added_to_cart = 1
            THEN session_id END)                                     AS cart_adds,
        COUNT(DISTINCT CASE WHEN checkout_started = 1
            THEN session_id END)                                     AS checkouts,
        COUNT(DISTINCT CASE WHEN order_placed = 1
            THEN session_id END)                                     AS orders
    FROM user_sessions
    WHERE session_date >= DATEADD(month, -3, GETDATE())
    GROUP BY acquisition_channel
)
SELECT
    acquisition_channel,
    sessions,
    product_views,
    ROUND(product_views * 100.0 / NULLIF(sessions, 0), 2)           AS view_rate_pct,
    cart_adds,
    ROUND(cart_adds * 100.0 / NULLIF(product_views, 0), 2)          AS cart_rate_pct,
    checkouts,
    ROUND(checkouts * 100.0 / NULLIF(cart_adds, 0), 2)              AS checkout_rate_pct,
    orders,
    ROUND(orders * 100.0 / NULLIF(sessions, 0), 2)                  AS overall_cvr_pct,
    -- Sessions that reached cart but didn't convert = recoverable margin signal
    (cart_adds - orders)                                             AS cart_abandonment_count
FROM funnel
ORDER BY overall_cvr_pct DESC;


-- 1b. Cart abandonment root cause — drop-off by stage and channel
-- Surfaces which channel and which stage is losing the most revenue
WITH abandonment AS (
    SELECT
        acquisition_channel,
        SUM(CASE WHEN added_to_cart = 1 AND order_placed = 0
            THEN cart_value ELSE 0 END)                              AS abandoned_revenue,
        COUNT(CASE WHEN added_to_cart = 1 AND order_placed = 0
            THEN session_id END)                                     AS abandoned_sessions,
        SUM(CASE WHEN added_to_cart = 1 AND checkout_started = 0
            THEN cart_value ELSE 0 END)                              AS lost_before_checkout,
        SUM(CASE WHEN checkout_started = 1 AND order_placed = 0
            THEN cart_value ELSE 0 END)                              AS lost_at_checkout
    FROM user_sessions
    WHERE session_date >= DATEADD(month, -3, GETDATE())
    GROUP BY acquisition_channel
)
SELECT
    acquisition_channel,
    ROUND(abandoned_revenue, 2)                                      AS total_abandoned_revenue,
    abandoned_sessions,
    ROUND(abandoned_revenue / NULLIF(abandoned_sessions, 0), 2)     AS avg_abandoned_cart_value,
    ROUND(lost_before_checkout, 2)                                   AS lost_pre_checkout,
    ROUND(lost_at_checkout, 2)                                       AS lost_at_checkout,
    -- Recoverable margin estimate (industry avg 15-20% winback rate)
    ROUND(abandoned_revenue * 0.15, 2)                               AS recoverable_margin_est
FROM abandonment
ORDER BY abandoned_revenue DESC;


-- ═══════════════════════════════════════════════════════════
-- SECTION 2: COHORT RETENTION ANALYSIS
-- Tracks monthly retention by customer acquisition cohort.
-- Identifies long-term engagement patterns invisible in
-- flat transaction reports.
-- ═══════════════════════════════════════════════════════════

-- 2a. Monthly cohort retention matrix
WITH first_purchase AS (
    -- Identify each customer's acquisition month
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(order_purchase_timestamp))          AS cohort_month
    FROM orders
    WHERE order_status = 'delivered'
    GROUP BY customer_id
),
order_months AS (
    -- Calculate how many months after acquisition each order was placed
    SELECT
        o.customer_id,
        fp.cohort_month,
        DATE_TRUNC('month', o.order_purchase_timestamp)             AS order_month,
        DATEDIFF(
            month,
            fp.cohort_month,
            DATE_TRUNC('month', o.order_purchase_timestamp)
        )                                                           AS months_since_acquisition
    FROM orders o
    JOIN first_purchase fp ON o.customer_id = fp.customer_id
    WHERE o.order_status = 'delivered'
),
cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id)                AS cohort_customers
    FROM first_purchase
    GROUP BY cohort_month
)
SELECT
    om.cohort_month,
    cs.cohort_customers,
    om.months_since_acquisition,
    COUNT(DISTINCT om.customer_id)                                  AS active_customers,
    ROUND(COUNT(DISTINCT om.customer_id) * 100.0
        / cs.cohort_customers, 2)                                   AS retention_rate_pct
FROM order_months om
JOIN cohort_size cs ON om.cohort_month = cs.cohort_month
GROUP BY om.cohort_month, cs.cohort_customers, om.months_since_acquisition
ORDER BY om.cohort_month, om.months_since_acquisition;


-- 2b. Repeat purchase rate by acquisition channel
-- Identifies which channels bring the highest-quality customers
SELECT
    acquisition_channel,
    COUNT(DISTINCT customer_id)                                     AS total_customers,
    COUNT(DISTINCT CASE WHEN order_count >= 2
        THEN customer_id END)                                       AS repeat_customers,
    ROUND(COUNT(DISTINCT CASE WHEN order_count >= 2
        THEN customer_id END) * 100.0
        / COUNT(DISTINCT customer_id), 2)                           AS repeat_purchase_rate_pct,
    ROUND(AVG(total_spend), 2)                                      AS avg_customer_ltv,
    ROUND(AVG(CASE WHEN order_count >= 2 THEN total_spend END), 2) AS repeat_customer_ltv
FROM (
    SELECT
        c.customer_id,
        c.acquisition_channel,
        COUNT(o.order_id)                                           AS order_count,
        SUM(oi.price + oi.freight_value)                            AS total_spend
    FROM customers c
    JOIN orders o     ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_id, c.acquisition_channel
) customer_summary
GROUP BY acquisition_channel
ORDER BY repeat_purchase_rate_pct DESC;


-- ═══════════════════════════════════════════════════════════
-- SECTION 3: A/B TEST FRAMEWORK
-- Statistically rigorous comparison of two discount strategies.
-- Includes sample ratio mismatch check and significance test.
-- ═══════════════════════════════════════════════════════════

-- 3a. Sample ratio mismatch check — run BEFORE looking at results
-- Uneven split invalidates the test before it starts
SELECT
    test_variant,
    COUNT(DISTINCT user_id)                                         AS users_assigned,
    COUNT(DISTINCT user_id) * 100.0
        / SUM(COUNT(DISTINCT user_id)) OVER ()                     AS pct_of_total,
    -- Flag if split deviates >5% from expected 50/50
    CASE
        WHEN ABS(COUNT(DISTINCT user_id) * 100.0
            / SUM(COUNT(DISTINCT user_id)) OVER () - 50) > 5
        THEN 'WARNING — Sample Ratio Mismatch'
        ELSE 'Split OK'
    END                                                             AS split_check
FROM ab_test_assignments
WHERE test_name = 'discount_strategy_q1_2024'
GROUP BY test_variant;


-- 3b. Primary metric comparison — conversion rate per variant
WITH variant_metrics AS (
    SELECT
        a.test_variant,
        COUNT(DISTINCT a.user_id)                                   AS users,
        COUNT(DISTINCT o.order_id)                                  AS conversions,
        COUNT(DISTINCT o.order_id) * 1.0
            / COUNT(DISTINCT a.user_id)                             AS conversion_rate,
        AVG(oi.price + oi.freight_value)                            AS avg_order_value,
        SUM(oi.price + oi.freight_value)                            AS total_revenue
    FROM ab_test_assignments a
    LEFT JOIN orders o
        ON a.user_id = o.customer_id
        AND o.order_purchase_timestamp BETWEEN a.assigned_at
            AND DATEADD(day, 14, a.assigned_at)
        AND o.order_status = 'delivered'
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE a.test_name = 'discount_strategy_q1_2024'
    GROUP BY a.test_variant
)
SELECT
    test_variant,
    users,
    conversions,
    ROUND(conversion_rate * 100, 4)                                 AS conversion_rate_pct,
    ROUND(avg_order_value, 2)                                       AS avg_order_value,
    ROUND(total_revenue, 2)                                         AS total_revenue,
    -- Relative uplift vs control (assumes control is variant A)
    ROUND((conversion_rate - FIRST_VALUE(conversion_rate)
        OVER (ORDER BY test_variant)) * 100
        / NULLIF(FIRST_VALUE(conversion_rate)
        OVER (ORDER BY test_variant), 0), 2)                        AS relative_uplift_pct
FROM variant_metrics
ORDER BY test_variant;


-- 3c. Statistical significance — Z-test for proportions
-- p < 0.05 required before making any business recommendation
WITH counts AS (
    SELECT
        test_variant,
        COUNT(DISTINCT a.user_id)                                   AS n,
        COUNT(DISTINCT o.order_id)                                  AS conversions
    FROM ab_test_assignments a
    LEFT JOIN orders o
        ON a.user_id = o.customer_id
        AND o.order_status = 'delivered'
    WHERE a.test_name = 'discount_strategy_q1_2024'
    GROUP BY test_variant
),
rates AS (
    SELECT
        MAX(CASE WHEN test_variant = 'A' THEN n END)                AS n_a,
        MAX(CASE WHEN test_variant = 'B' THEN n END)                AS n_b,
        MAX(CASE WHEN test_variant = 'A'
            THEN conversions * 1.0 / n END)                         AS p_a,
        MAX(CASE WHEN test_variant = 'B'
            THEN conversions * 1.0 / n END)                         AS p_b
    FROM counts
)
SELECT
    ROUND(p_a * 100, 4)                                             AS conversion_rate_a_pct,
    ROUND(p_b * 100, 4)                                             AS conversion_rate_b_pct,
    ROUND((p_b - p_a) * 100, 4)                                     AS absolute_lift_pct,
    -- Pooled proportion for Z-test
    ROUND((p_a * n_a + p_b * n_b) / (n_a + n_b), 6)               AS pooled_p,
    -- Z-score: positive = B outperforms A
    ROUND(
        (p_b - p_a) / SQRT(
            ((p_a * n_a + p_b * n_b) / (n_a + n_b)) *
            (1 - (p_a * n_a + p_b * n_b) / (n_a + n_b)) *
            (1.0 / n_a + 1.0 / n_b)
        ), 4
    )                                                               AS z_score,
    CASE
        WHEN ABS((p_b - p_a) / SQRT(
            ((p_a * n_a + p_b * n_b) / (n_a + n_b)) *
            (1 - (p_a * n_a + p_b * n_b) / (n_a + n_b)) *
            (1.0 / n_a + 1.0 / n_b))) >= 1.96
        THEN 'Statistically Significant (p < 0.05)'
        ELSE 'Not Significant — Do Not Ship'
    END                                                             AS significance
FROM rates;


-- ═══════════════════════════════════════════════════════════
-- SECTION 4: OPERATIONAL KPI DASHBOARD QUERIES
-- Weekly performance summary designed to connect to Tableau.
-- ═══════════════════════════════════════════════════════════

-- 4a. Weekly revenue, order volume, and AOV
SELECT
    DATE_TRUNC('week', order_purchase_timestamp)                    AS week_start,
    COUNT(DISTINCT o.order_id)                                      AS total_orders,
    COUNT(DISTINCT o.customer_id)                                   AS unique_customers,
    ROUND(SUM(oi.price + oi.freight_value), 2)                     AS gross_revenue,
    ROUND(AVG(oi.price + oi.freight_value), 2)                     AS avg_order_value,
    -- Week-over-week revenue growth
    ROUND(
        (SUM(oi.price + oi.freight_value) - LAG(SUM(oi.price + oi.freight_value))
            OVER (ORDER BY DATE_TRUNC('week', order_purchase_timestamp)))
        * 100.0 / NULLIF(LAG(SUM(oi.price + oi.freight_value))
            OVER (ORDER BY DATE_TRUNC('week', order_purchase_timestamp)), 0),
        2
    )                                                               AS wow_revenue_growth_pct
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp >= DATEADD(month, -6, GETDATE())
GROUP BY DATE_TRUNC('week', order_purchase_timestamp)
ORDER BY week_start DESC;


-- 4b. Delivery performance by seller — feeds vendor KPI dashboard
SELECT
    oi.seller_id,
    COUNT(DISTINCT o.order_id)                                      AS total_orders,
    ROUND(AVG(DATEDIFF(day,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date)), 1)                       AS avg_delivery_days,
    SUM(CASE WHEN o.order_delivered_customer_date
            <= o.order_estimated_delivery_date THEN 1 ELSE 0 END)  AS on_time_deliveries,
    ROUND(SUM(CASE WHEN o.order_delivered_customer_date
            <= o.order_estimated_delivery_date THEN 1 ELSE 0 END)
        * 100.0 / COUNT(DISTINCT o.order_id), 2)                   AS on_time_delivery_pct,
    ROUND(AVG(r.review_score), 2)                                   AS avg_review_score
FROM order_items oi
JOIN orders o         ON oi.order_id = o.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY oi.seller_id
HAVING COUNT(DISTINCT o.order_id) >= 10   -- Minimum volume filter
ORDER BY on_time_delivery_pct ASC;        -- Worst performers first
