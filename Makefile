default: 1-grille.html  2-animation_in_a_notebook.html  3-adaptation.html 4-vitesse.html 5-distance.html 6-croix.html

%.html: %.ipynb
	runipy $< -o
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to html $<

%.pdf: %.ipynb
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to latex --post PDF $<

# cleaning macros
clean:
	#find .  -name *lock* -exec rm -fr {} \;
	rm -fr  *.html *.pyc *.py~

.PHONY: clean
