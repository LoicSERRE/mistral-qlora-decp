# Configuration latexmk pour le projet ADS

# Utiliser pdflatex par défaut
$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Compiler automatiquement les bibliographies
$bibtex_use = 2;

# Nombre maximum de compilations
$max_repeat = 5;

# Options pdflatex
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Nettoyer les fichiers auxiliaires
@generated_exts = qw(aux bbl blg fdb_latexmk fls log out synctex.gz toc);

# Sortie dans le même répertoire
$out_dir = '.';

# Preview en continu (optionnel)
# $preview_continuous_mode = 1;
