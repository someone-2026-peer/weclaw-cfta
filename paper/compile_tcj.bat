@echo off
REM Build the CFTA -> The Computer Journal (OUP) submission PDF.
REM Engine: XeLaTeX (required for xeCJK / the Chinese query example).
REM Two passes so that cross-references and the numbered bibliography settle.
cd /d "%~dp0"
xelatex -interaction=nonstopmode paper_tcj.tex
xelatex -interaction=nonstopmode paper_tcj.tex
echo.
echo Done. Output: paper_tcj.pdf
