@echo off
cd /d "C:\Pesquisas\PMSP_Learnheuristic"
echo Iniciando run_ils_hi_new_classes_entrada70.py...
echo Estimativa: ~2,4h (ver docstring do script para detalhes)
echo.
where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python experiment\run_ils_hi_new_classes_entrada70.py
) else (
    py experiment\run_ils_hi_new_classes_entrada70.py
)
echo.
echo ===== CONCLUIDO (ou interrompido -- ver mensagens acima) =====
pause
