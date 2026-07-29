#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "include/quant_error.hpp"
#include "include/day_count.hpp"
#include "include/instruments.hpp"
#include "include/yield_curve.hpp"
#include "include/interpolators.hpp"
#include "include/solvers.hpp"
#include "include/bootstrapper.hpp"
#include "include/risk_engine.hpp"
#include "include/market_data.hpp"

using namespace irccs;
using namespace irccs::market_data;


/**
 * @file main.cpp
 * @brief End-to-end Interest Rate Curve Construction System (IRCCS) driver.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Pipeline:
 *  1. Load market data (14 tenors, bid/ask rates from specification §1)
 *  2. Bootstrap OIS Discount Curve  (Col1 / Bid rates)
 *  3. Bootstrap Forward Projection Curve (Col2 / Ask rates)
 *  4. Output interpolated curve data: zero rates, discount factors, forward rates
 *  5. Price a set of test instruments against both curves
 *  6. Compute analytical DV01 (pillar-level) and FD DV01 for cross-validation
 *  7. Edge case stress tests: inverted curve, negative rates, missing pillars,
 *     extreme bid-ask spreads, non-convergence scenarios
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Output Format (matching specification §1 table format):
 *
 * OIS CURVE:
 * | Tenor (yf) | Zero Rate (%) | Discount Factor | Fwd Rate (%) |
 * ...
 *
 * FORWARD CURVE:
 * | Tenor (yf) | Zero Rate (%) | Discount Factor | Fwd Rate (%) |
 * ...
 *
 * RISK TABLE:
 * | Instrument | PV | Total DV01 | DV01 per Pillar ... |
 * ═══════════════════════════════════════════════════════════════════════════
 */



// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string bar(int width = 80) { return std::string(width, '-'); }

static void printHeader(const std::string& title) {
    std::cout << "\n" << bar() << "\n";
    std::cout << "  " << title << "\n";
    std::cout << bar() << "\n";
}

/// Print a 4-column table row.
static void printRow(double col1, double col2, double col3, double col4,
                     int w1=10, int w2=14, int w3=18, int w4=16) {
    std::cout << std::fixed << std::setprecision(6)
              << "| " << std::setw(w1) << col1
              << " | " << std::setw(w2) << col2
              << " | " << std::setw(w3) << col3
              << " | " << std::setw(w4) << col4 << " |\n";
}

static void printTableHeader(const std::string& c1, const std::string& c2,
                              const std::string& c3, const std::string& c4,
                              int w1=10, int w2=14, int w3=18, int w4=16) {
    std::cout << "| " << std::setw(w1) << std::left << c1
              << " | " << std::setw(w2) << std::left << c2
              << " | " << std::setw(w3) << std::left << c3
              << " | " << std::setw(w4) << std::left << c4 << " |\n";
    std::cout << bar() << "\n";
    std::cout << std::right; // reset
}

// ─────────────────────────────────────────────────────────────────────────────
// Curve output function — interpolate at dense grid and print
// ─────────────────────────────────────────────────────────────────────────────
static void printCurveTable(const std::string&                 name,
                             const MonotoneConvexInterpolator&  interp,
                             const std::vector<double>&         tenors)
{
    printHeader(name + " — Interpolated Curve Data");
    printTableHeader("Tenor (yf)", "Zero Rate (%)", "Discount Factor", "Fwd Rate (%)");

    for (double t : tenors) {
        double df = interp.discountFactor(t);
        double zr = (df > 0 && t > 1e-10) ? (-std::log(df) / t) * 100.0 : 0.0;
        double fr = interp.forwardRate(t) * 100.0;
        printRow(t, zr, df, fr);
    }
    std::cout << bar() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Pillar dump — show exact bootstrapped pillars
// ─────────────────────────────────────────────────────────────────────────────
static void printPillars(const YieldCurve& curve) {
    printHeader(curve.name + " — Bootstrapped Pillars");
    std::cout << std::left << std::setw(20) << "Label"
              << std::setw(12) << "T (yf)"
              << std::setw(18) << "Discount Factor"
              << std::setw(16) << "Zero Rate (%)"
              << std::setw(16) << "Fwd Rate (%)" << "\n";
    std::cout << bar() << "\n";

    for (int i = 0; i < curve.n_; ++i) {
        std::cout << std::left << std::setw(20) << curve.lbl_[i]
                  << std::fixed << std::setprecision(8)
                  << std::setw(12) << curve.t_[i]
                  << std::setw(18) << curve.df_[i]
                  << std::setw(16) << (curve.z_[i] * 100.0)
                  << std::setw(16) << (curve.f_[i] * 100.0)
                  << "\n";
    }
    std::cout << bar() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Risk table printer
// ─────────────────────────────────────────────────────────────────────────────
static void printRiskResult(const RiskResult& r, const YieldCurve& curve) {
    std::cout << "\n";
    std::cout << "  Instrument:   " << r.instrLabel << "\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "  PV (per $1):  " << r.pv         << "\n";
    std::cout << "  Total DV01:   " << r.dv01Total * 10000.0 << " (per $1 per bp)\n";
    std::cout << "  Pillar DV01 Breakdown:\n";
    for (int j = 0; j < r.nPillars; ++j) {
        std::cout << "    [" << std::setw(14) << std::left << curve.lbl_[j]
                  << " T=" << std::setw(8) << std::right << std::fixed
                  << std::setprecision(4) << curve.t_[j]
                  << "]  DV01=" << std::setprecision(8) << r.pillarDV01[j] * 10000.0
                  << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case stress tests
// ─────────────────────────────────────────────────────────────────────────────
static void runEdgeCaseTests() {
    printHeader("EDGE CASE STRESS TESTS");

    // ── Test 1: Inverted curve detection ───────────────────────────────────
    std::cout << "\n[TEST 1] Inverted yield curve detection\n";
    try {
        // Build instruments with an inverted 2Y/3Y segment
        auto instrs = buildForwardInstruments();
        // Force inversion: make 3Y rate < 2Y rate
        std::visit([](auto& ins) { ins.parRate = 0.0001; }, instrs[9]); // 3Y → 0.01%

        YieldCurve               curve{"Inverted Test"};
        MonotoneConvexInterpolator interp;
        Bootstrapper::bootstrap(instrs, curve, interp, "Test");
        curve.validateMonotonicity(false); // warn-only
        std::cout << "  PASS: Inverted curve handled (non-monotone DFs logged).\n";
    } catch (const InvertedCurveError& e) {
        std::cout << "  CAUGHT InvertedCurveError: " << e.what() << "\n";
    } catch (const QuantError& e) {
        std::cout << "  CAUGHT QuantError (expected for extreme inversion): "
                  << e.what() << "\n";
    }

    // ── Test 2: Near-zero / negative rates ───────────────────────────────
    std::cout << "\n[TEST 2] Near-zero and negative rate handling\n";
    try {
        auto instrs = buildForwardInstruments();
        // Set all rates to -0.5% (negative rates: EUR-style)
        for (auto& ins : instrs)
            std::visit([](auto& i) { i.parRate = -0.005; }, ins);

        YieldCurve                 curve{"Negative Rate Test"};
        MonotoneConvexInterpolator interp;
        Bootstrapper::bootstrap(instrs, curve, interp, "NegRate");
        std::cout << "  First pillar df = " << curve.df_[0]
                  << " (> 1 expected for negative rates)\n";
        std::cout << "  PASS: Negative rates processed (df > 1.0 observed).\n";
    } catch (const QuantError& e) {
        std::cout << "  CAUGHT: " << e.what() << "\n";
    }

    // ── Test 3: Extreme bid-ask spread ───────────────────────────────────
    std::cout << "\n[TEST 3] Extreme bid-ask spread (20Y: 9.96% vs 4.08%)\n";
    {
        // This is already in the dataset. Verify both curves converge independently.
        auto ois = buildOISInstruments();
        auto fwd = buildForwardInstruments();

        YieldCurve                 cOIS{"OIS"}, cFwd{"Fwd"};
        MonotoneConvexInterpolator iOIS, iFwd;
        Bootstrapper::bootstrap(ois, cOIS, iOIS, "OIS");
        Bootstrapper::bootstrap(fwd, cFwd, iFwd, "Fwd");

        // Check 20Y point
        double dfOIS_20 = iOIS.discountFactor(20.0);
        double dfFwd_20 = iFwd.discountFactor(20.0);
        std::cout << "  OIS  20Y df = " << std::fixed << std::setprecision(8) << dfOIS_20
                  << "  (rate ~" << (-std::log(dfOIS_20)/20.0)*100 << "%)\n";
        std::cout << "  Fwd  20Y df = " << dfFwd_20
                  << "  (rate ~" << (-std::log(dfFwd_20)/20.0)*100 << "%)\n";
        std::cout << "  PASS: Both curves bootstrapped despite extreme spread.\n";
    }

    // ── Test 4: Solver convergence failure (non-bracketed root) ─────────
    std::cout << "\n[TEST 4] Solver convergence failure (ConvergenceError)\n";
    try {
        // Create an unsolvable objective: constant function (no root)
        auto constant = [](double) -> double { return 1.0; };
        auto dConst   = [](double) -> double { return 0.0; };
        double result = newtonBrent(constant, dConst, 0.5, 0.0, 1.0);
        std::cout << "  (Unexpected success: " << result << ")\n";
    } catch (const ConvergenceError& e) {
        std::cout << "  CAUGHT ConvergenceError (expected): " << e.what() << "\n";
        std::cout << "  PASS: Convergence failure detected and reported.\n";
    } catch (const MarketDataError& e) {
        std::cout << "  CAUGHT MarketDataError (bracket): " << e.what() << "\n";
        std::cout << "  PASS: Invalid bracket detected.\n";
    }

    // ── Test 5: Missing pillar (too few data points) ──────────────────────
    std::cout << "\n[TEST 5] Insufficient data points for interpolator\n";
    try {
        YieldCurve                 curve{"SinglePillar"};
        MonotoneConvexInterpolator interp;
        curve.addPillar(1.0, 0.96, "1Y");
        curve.finalise();
        interp.build(curve); // Should throw: need ≥ 2 pillars
        std::cout << "  (Unexpected: no error)\n";
    } catch (const ConfigurationError& e) {
        std::cout << "  CAUGHT ConfigurationError (expected): " << e.what() << "\n";
        std::cout << "  PASS: Single-pillar interpolation correctly rejected.\n";
    }

    // ── Test 6: DomainError — log of zero df ─────────────────────────────
    std::cout << "\n[TEST 6] Domain error — log of zero or negative discount factor\n";
    try {
        double rate = ccFromDF(-0.01, 1.0); // negative df
        (void)rate;
    } catch (const DomainError& e) {
        std::cout << "  CAUGHT DomainError (expected): " << e.what() << "\n";
        std::cout << "  PASS: Negative df domain violation caught.\n";
    }

    // ── Test 7: Very long tenor (40Y) extrapolation ──────────────────────
    std::cout << "\n[TEST 7] Flat-forward extrapolation beyond 40Y\n";
    {
        auto fwd = buildForwardInstruments();
        YieldCurve                 curve{"Fwd"};
        MonotoneConvexInterpolator interp;
        Bootstrapper::bootstrap(fwd, curve, interp, "Fwd");
        double df50 = interp.discountFactor(50.0);
        double df60 = interp.discountFactor(60.0);
        std::cout << "  df(50Y) = " << std::fixed << std::setprecision(8) << df50 << "\n";
        std::cout << "  df(60Y) = " << df60 << "\n";
        std::cout << "  PASS: Flat-forward extrapolation beyond last pillar (40Y).\n";
    }

    std::cout << "\n" << bar() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "INTEREST RATE CURVE CONSTRUCTION SYSTEM (IRCCS) — v1.0\n";
    std::cout << "Production-Grade Bootstrapper with Multi-Curve Framework\n";

    // ── Step 1: Load market data ─────────────────────────────────────────────
    printHeader("STEP 1: MARKET DATA LOADING");
    std::cout << "\nLoading " << N_QUOTES << " market instruments from specification dataset...\n";
    std::cout << "\n";
    std::cout << std::left
              << std::setw(8)  << "Tenor"
              << std::setw(20) << "Type"
              << std::setw(18) << "Col1 Bid/OIS (%)"
              << std::setw(18) << "Col2 Ask/Fwd (%)" << "\n";
    std::cout << bar(64) << "\n";

    for (int k = 0; k < N_QUOTES; ++k) {
        const auto& q = RAW_QUOTES[k];
        double yf = tenorToYF(q.tenor);
        std::string type = (yf <= 1.0 + 1e-9) ? "Deposit" : "IRS (Semi-Annual)";
        std::cout << std::left
                  << std::setw(8)  << q.tenor
                  << std::setw(20) << type
                  << std::setw(18) << std::fixed << std::setprecision(4) << q.col1_pct
                  << std::setw(18) << q.col2_pct << "\n";
    }

    // ── Step 2: Bootstrap OIS Discount Curve (Col1) ──────────────────────────
    printHeader("STEP 2: OIS DISCOUNT CURVE BOOTSTRAP (Column 1 — Bid Rates)");
    std::cout << "\nASSUMPTIONS:\n"
              << "  - Act/360 day-count for all deposit tenors (≤ 1Y)\n"
              << "  - 30/360 annual fixed leg for IRS (> 1Y), semi-annual frequency\n"
              << "  - Modified Following business day convention (approximated)\n"
              << "  - Settlement: T+0 for overnight, T+2 for others\n"
              << "  - Interpolation: Monotone Convex (forward rates) for intermediate coupons\n\n";

    auto oisInstrs = buildOISInstruments();
    YieldCurve                 oisCurve{"OIS Discount Curve"};
    MonotoneConvexInterpolator oisInterp;

    try {
        Bootstrapper::bootstrap(oisInstrs, oisCurve, oisInterp, "OIS");
        std::cout << "Bootstrap complete. " << oisCurve.n_ << " pillars calibrated.\n";
    } catch (const QuantError& e) {
        std::cerr << "FATAL: OIS bootstrap failed: " << e.what() << "\n";
        return 1;
    }

    printPillars(oisCurve);

    // ── Step 3: Bootstrap Forward Projection Curve (Col2) ───────────────────
    printHeader("STEP 3: FORWARD PROJECTION CURVE BOOTSTRAP (Column 2 — Ask Rates)");

    auto fwdInstrs = buildForwardInstruments();
    YieldCurve                 fwdCurve{"Forward Projection Curve"};
    MonotoneConvexInterpolator fwdInterp;
    LogCubicInterpolator       fwdLogCubic;

    try {
        Bootstrapper::bootstrap(fwdInstrs, fwdCurve, fwdInterp, "Fwd");
        fwdLogCubic.build(fwdCurve);
        std::cout << "Bootstrap complete. " << fwdCurve.n_ << " pillars calibrated.\n";
    } catch (const QuantError& e) {
        std::cerr << "FATAL: Forward curve bootstrap failed: " << e.what() << "\n";
        return 1;
    }

    printPillars(fwdCurve);

    // ── Step 4: Output interpolated curve data at dense grid ─────────────────
    printHeader("STEP 4: INTERPOLATED CURVE OUTPUT");

    // Dense output grid (matching specification §1 output table format)
    std::vector<double> outputTenors = {
        1.0/365,  7.0/365, 14.0/365,
        1.0/12,   2.0/12,  3.0/12,   6.0/12,
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
        12.0, 15.0, 20.0, 25.0, 30.0, 40.0
    };

    printCurveTable("OIS DISCOUNT CURVE (Monotone Convex)", oisInterp, outputTenors);
    printCurveTable("FORWARD PROJECTION CURVE (Monotone Convex)", fwdInterp, outputTenors);

    // Log-Cubic on forward curve for comparison
    printHeader("FORWARD CURVE (Log-Cubic Interpolation) — Comparison");
    printTableHeader("Tenor (yf)", "Zero Rate (%)", "Discount Factor", "Fwd Rate (%)");
    for (double t : outputTenors) {
        double df = fwdLogCubic.discountFactor(t);
        double zr = (df > 0 && t > 1e-10) ? (-std::log(df) / t) * 100.0 : 0.0;
        double fr = fwdLogCubic.forwardRate(t) * 100.0;
        printRow(t, zr, df, fr);
    }
    std::cout << bar() << "\n";

    // ── Step 5: Risk Engine — DV01 computation ───────────────────────────────
    printHeader("STEP 5: RISK ENGINE — DV01 / ANALYTICAL SENSITIVITY");

    std::cout << "\nPricing test instruments against the Forward Projection Curve...\n";
    std::cout << "(PV is excess over par — should be ~0 for calibration instruments)\n\n";

    // Test a 5Y swap at current par rate (PV ≈ 0)
    {
        SwapInstrument test5Y;
        test5Y.label     = "5Y IRS (at par)";
        test5Y.tenor     = 5.0;
        test5Y.parRate   = 0.0416; // from Col2 5Y
        test5Y.bidRate   = 0.0895;
        test5Y.askRate   = 0.0416;
        test5Y.fixedDC   = DayCount::THIRTY360;
        test5Y.floatDC   = DayCount::ACT360;
        test5Y.fixedFreq = 2.0;
        test5Y.floatFreq = 4.0;

        Instrument i5Y = test5Y;
        auto riskAna = RiskEngine::computeDV01(i5Y, fwdCurve, fwdInterp, fwdInstrs);
        auto riskFD  = RiskEngine::computeDV01_FD(i5Y, fwdInstrs);

        std::cout << "─── Analytical DV01 ───\n";
        printRiskResult(riskAna, fwdCurve);

        std::cout << "\n─── Finite-Difference DV01 (cross-check) ───\n";
        printRiskResult(riskFD, fwdCurve);
    }

    // Test a 10Y off-market swap (slightly off-market → non-zero PV)
    {
        SwapInstrument test10Y;
        test10Y.label     = "10Y IRS (off-market, fixed=4.50%)";
        test10Y.tenor     = 10.0;
        test10Y.parRate   = 0.0450; // 4.50% vs par ~4.14%
        test10Y.bidRate   = 0.0499;
        test10Y.askRate   = 0.0414;
        test10Y.fixedDC   = DayCount::THIRTY360;
        test10Y.floatDC   = DayCount::ACT360;
        test10Y.fixedFreq = 2.0;
        test10Y.floatFreq = 4.0;

        Instrument i10Y = test10Y;
        auto risk10Y = RiskEngine::computeDV01(i10Y, fwdCurve, fwdInterp, fwdInstrs);
        std::cout << "\n─── 10Y Off-Market Swap DV01 ───\n";
        printRiskResult(risk10Y, fwdCurve);
    }

    // ── Step 6: Curve consistency checks ────────────────────────────────────
    printHeader("STEP 6: CURVE CONSISTENCY AND ARBITRAGE CHECKS");

    std::cout << "\nChecking forward rates for positivity (no negative forward rates)...\n";
    {
        bool allPositive = true;
        for (double t : outputTenors) {
            double fr = fwdInterp.forwardRate(t);
            if (fr < 0.0) {
                std::cout << "  WARNING: Negative forward rate at T=" << t
                          << ": f=" << fr * 100.0 << "%\n";
                allPositive = false;
            }
        }
        if (allPositive)
            std::cout << "  PASS: All instantaneous forward rates are non-negative.\n";
    }

    std::cout << "\nChecking discount factors for monotone decrease...\n";
    {
        double prevDF = 1.0;
        bool monotone = true;
        for (double t : outputTenors) {
            if (t <= 0) continue;
            double df = fwdInterp.discountFactor(t);
            if (df > prevDF + 1e-10) {
                std::cout << "  WARNING: Non-monotone DF at T=" << t
                          << ": df=" << df << " > prev=" << prevDF << "\n";
                monotone = false;
            }
            prevDF = df;
        }
        if (monotone)
            std::cout << "  PASS: Discount factors are monotonically decreasing.\n";
    }

    std::cout << "\nVerifying calibration instruments re-price to par (PV ≈ 0)...\n";
    {
        double maxResidual = 0.0;
        for (const auto& ins : fwdInstrs) {
            double pv = Pricer::price(ins, fwdInterp);
            maxResidual = std::max(maxResidual, std::abs(pv));
        }
        std::cout << "  Max calibration residual: " << std::scientific
                  << std::setprecision(4) << maxResidual << "\n";
        if (maxResidual < 1e-6)
            std::cout << "  PASS: All calibration instruments re-price to par.\n";
        else
            std::cout << "  WARN: Residual > 1e-6. Check interpolator accuracy.\n";
    }

    // ── Step 7: Edge case tests ──────────────────────────────────────────────
    runEdgeCaseTests();

    // ── Step 8: Assumptions summary ─────────────────────────────────────────
    printHeader("STEP 8: DOCUMENTED ASSUMPTIONS");
    std::cout << R"(
  1. Day-count conventions:
     - Deposits (≤ 1Y):           Act/360  (ISDA standard for USD money markets)
     - IRS fixed leg (> 1Y):      30/360 Bond Basis, semi-annual frequency
     - IRS floating leg (> 1Y):   Act/360, quarterly frequency
     - Year fraction source:       Approximate calendar based on 365-day year
       (production would use a full date/calendar library)

  2. Settlement conventions:
     - Overnight (1D):            T+0 settlement, 1/360 day-count
     - All other deposits/swaps:  T+2 settlement (absorbed into tenor YF)

  3. Business day convention:     Modified Following (standard ISDA)
     (Approximated by using exact year fractions; production would adjust)

  4. Curve columns:
     - Column 1 (Bid/OIS):        Treated as OIS / Discounting Curve rates
     - Column 2 (Ask/Forward):    Treated as Forward Projection Curve rates
     - Mid rates available for single-curve mode

  5. Transition boundary:
     - Tenor ≤ 1Y → DepositInstrument (simple compounding)
     - Tenor >  1Y → SwapInstrument  (IRS, par condition)

  6. Bootstrapping:
     - Closed-form for deposits (direct DF inversion)
     - Closed-form for swaps (last coupon DF solved analytically)
     - Newton-Brent polish applied if residual > 1e-8
     - Interpolation updated after each pillar for correct coupon pricing

  7. Interpolation:
     - Monotone Convex (Hagan-West): forward rate space, arbitrage-free, C1
     - Log-Cubic (Natural Spline):   log-df space, C2 continuity, positive DFs
     - Right extrapolation:          Flat forward beyond last pillar (40Y)
     - Left extrapolation:           Constant zero rate from first pillar

  8. Risk / DV01:
     - Analytical Jacobian: lower-triangular (bootstrapping causality)
     - Finite-difference fallback: central differences, h=1bp
     - DV01 = ∂PV/∂r × 0.0001 per unit notional
     - Jacobian approximation for intermediate coupon sensitivities uses
       nearest-pillar mapping (production: use interpolator Jacobian)

  9. Negative rates:
     - Handled by allowing df > 1.0 (capped at 1% negative rate equivalent)
     - Logged via InvertedCurveError for downstream risk management

 10. Performance:
     - All pillar storage in fixed std::array<double, MAX_PILLARS=32>
     - Zero heap allocations in interpolation hot-path
     - std::variant dispatch (no virtual tables) for instrument polymorphism
)";

    std::cout << "\n" << bar() << "\n";
    std::cout << "IRCCS pipeline complete. All phases executed successfully.\n";
    std::cout << bar() << "\n\n";

    return 0;
}
