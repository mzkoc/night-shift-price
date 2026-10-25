# Parameter tuning of the rule-seeded BRKGA (BRKGA-S) with irace, same protocol as GP-RKO:
# ten separate class-L tuning instances (seeds 101-110), Lex10 scheme, night rule,
# 100000 evaluations per run, 400 runs.  Run from the folder with weaving_gp_v8.py (C:\weaving_gp):
#   1. the payoff tables tune_payoff_101.json ... tune_payoff_110.json must exist
#      (they were made for GP-RKO; otherwise: python -u weaving_gp_v8.py --tune-prep)
#   2. setwd("C:/weaving_gp"); source("irace_tune_brkga.R")     (RStudio)
# The best configuration is written to brkga_s_cfg.json, which weaving_gp_v8.py reads at start-up.
library(irace)

parameters_table <- '
p    "--p "    i (30, 200)
pe   "--pe "   r (0.05, 0.35)
pm   "--pm "   r (0.05, 0.30)
rho  "--rho "  r (0.50, 0.90)
'
parameters <- readParameters(text = parameters_table)

target_runner <- function(experiment, scenario) {
  py   <- "python"                                   # change to the full path of python.exe if needed
  conf <- experiment$configuration
  args <- c("weaving_gp_v8.py", "--irace-brkga", as.character(experiment$instance), as.character(experiment$seed))
  for (p in names(conf)) {
    v <- conf[[p]]
    if (!is.na(v)) {
      v <- if (p == "p") as.character(as.integer(v)) else sprintf("%.4f", as.numeric(v))
      args <- c(args, paste0("--", p), v)
    }
  }
  out <- system2(py, args, stdout = TRUE, stderr = FALSE)
  list(cost = as.numeric(utils::tail(out, 1)))
}

# standard BRKGA settings as the first candidate
default_conf <- data.frame(p = 100L, pe = 0.20, pm = 0.15, rho = 0.70)

scenario <- list(targetRunner   = target_runner,
                 instances      = as.character(101:110),
                 maxExperiments = 400,
                 parallel       = 18,                 # set to 1 if parallel runs fail on your system
                 initConfigurations = default_conf,
                 logFile        = "irace_v7_brkga.Rdata",
                 parameters     = parameters)

tuned <- irace(scenario = scenario)
best  <- removeConfigurationsMetaData(tuned)
print(tuned)
cat("\nBest configurations (first row = best):\n"); print(best)
b <- best[1, ]
writeLines(sprintf('{"p": %d, "pe": %.4f, "pm": %.4f, "rho": %.4f}', as.integer(b$p), b$pe, b$pm, b$rho),
           "brkga_s_cfg.json")
cat("\nWritten brkga_s_cfg.json\n")
