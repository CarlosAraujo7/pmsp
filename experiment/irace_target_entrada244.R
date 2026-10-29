# experiment/irace_target_entrada244.R
# Entrada 244 (Ciclo 24): o irace (Lopez-Ibanez et al. 2016; pacote R irace) como configurador da acao ESTATICA e dos hiperparametros do Q-Learning. O alvo roda em Python (numba): a cada experimento o irace manda uma linha
# "rep <TAB> semente <TAB> p1 <TAB> p2 <TAB> p3 <TAB> p4" a um servidor local (experiment/run_cycle24_irace_entrada244.py) e le' o custo (lacuna ao limite inferior, em %) de volta.
# Uso: Rscript experiment/irace_target_entrada244.R PORTA TIPO ORCAMENTO SEMENTE SAIDA.csv N_INSTANCIAS     (TIPO: cfg5, cfg15 ou q; biblioteca R em R_LIBS_USER)
# Tudo no padrao do irace (4.5), exceto maxExperiments e a semente.
lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages(library(irace))
args <- commandArgs(trailingOnly = TRUE)
port <- as.integer(args[1]); kind <- args[2]; budget <- as.integer(args[3]); seed <- as.integer(args[4]); out <- args[5]; ninst <- as.integer(args[6])

ptxt <- switch(kind,
  cfg5 = 'dfrac "" r (0.05, 0.40)\n',
  cfg15 = 'strat "" c (0, 1, 2)\ndfrac "" r (0.05, 0.40)\n',
  q = 'alpha "" r,log (0.01, 0.9)\ngamma "" r (0.0, 0.99)\ndecay "" r (0.98, 0.9999)\nfloor "" r (0.0, 0.5)\n',
  stop("tipo desconhecido: ", kind))
parameters <- readParameters(text = ptxt)

con <- socketConnection(host = "127.0.0.1", port = port, blocking = TRUE, open = "r+", timeout = 600)
num <- function(x) sprintf("%.10g", as.numeric(x))
runner <- function(experiment, scenario) {
  cfg <- experiment$configuration
  p <- switch(kind,
    cfg5 = c(0, cfg[["dfrac"]], 0, 0),
    cfg15 = c(cfg[["strat"]], cfg[["dfrac"]], 0, 0),
    q = c(cfg[["alpha"]], cfg[["gamma"]], cfg[["decay"]], cfg[["floor"]]))
  rep <- as.integer(sub("^i", "", experiment$instance))
  writeLines(paste(c(rep, experiment$seed, vapply(p, num, "")), collapse = "\t"), con)
  flush(con)
  ans <- readLines(con, n = 1)
  list(cost = as.numeric(ans))
}

scenario <- list(targetRunner = runner, instances = paste0("i", 0:(ninst - 1)), maxExperiments = budget, seed = seed,
                 parameters = parameters, logFile = "", debugLevel = 0, parallel = 1)
res <- irace(scenario = scenario)
write.csv(res, out, row.names = FALSE)
close(con)
