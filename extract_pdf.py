import fitz

doc = fitz.open(r'd:\Projets\ADS\docs\Subject\Guide-Redaction-ADS.pdf')
for page in doc:
    print(page.get_text())
