# Parameter tuning of GP-RKO (done in v6, configuration 47; kept for reproducibility) with irace (separate tuning instances, class L, seeds 101-110).
# Run from the folder that contains weaving_gp_v8.py (C:\weaving_gp):
#   1. python -u weaving_gp_v8.py --tune-prep      (payoff tables of the 10 tuning instances)
#   2. in R, once: install.packages("irace")
#   3. Rscript irace_tune.R                         (or setwd("C:/weaving_gp"); source("irace_tune.R") in RStudio)
# Each target run is one GP-RKO run (100000 evaluations, about 1 minute) on one tuning instance.
library(irace)

parameters_table <- '
psize    "--psize "    i (10, 60)
xbias    "--xbias "    r (0.50, 0.90)
sa_moves "--sa_moves " i (5, 30)
sa_T0    "--sa_T0 "    r (0.05, 1.00)
sa_cool  "--sa_cool "  r (0.70, 0.98)
ils_len  "--ils_len "  i (5, 40)
pmerge   "--pmerge "   r (0.00, 0.40)
'
parameters <- readParameters(text = parameters_table)

target_runner <- function(experiment, scenario) {
  py   <- "python"                                   # change to the full path of python.exe if needed
  conf <- experiment$configuration
  args <- c("weaving_gp_v8.py", "--irace", as.character(experiment$instance), as.character(experiment$seed))
  for (p in names(conf)) {
    v <- conf[[p]]
    if (!is.na(v)) {
      v <- if (p %in% c("psize", "sa_moves", "ils_len")) as.character(as.integer(v)) else sprintf("%.4f", as.numeric(v))
      args <- c(args, paste0("--", p), v)
    }
  }
  out <- system2(py, args, stdout = TRUE, stderr = FALSE)
  list(cost = as.numeric(utils::tail(out, 1)))
}

# default configuration as the first candidate
default_conf <- data.frame(psize = 30L, xbias = 0.70, sa_moves = 10L, sa_T0 = 0.30, sa_cool = 0.85,
                           ils_len = 15L, pmerge = 0.15)

scenario <- list(targetRunner   = target_runner,
                 instances      = as.character(101:110),
                 maxExperiments = 400,
                 parallel       = 18,                 # set to 1 if parallel runs fail on your system
                 initConfigurations = default_conf,
                 logFile        = "irace_v7_rko.Rdata",
                 parameters     = parameters)

tuned <- irace(scenario = scenario)
print(tuned)
cat("\nBest configurations (first row = best):\n")
print(removeConfigurationsMetaData(tuned))
