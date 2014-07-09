default: 1-grille.html  2-animation_in_a_notebook.html  3-adaptation.html

%.html: %.ipynb
	runipy $< --html $@

%.pdf: %.ipynb
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to latex --post PDF $<

# cleaning macros
clean:
	find .  -name *lock* -exec rm -fr {} \;
	rm -fr figures/* *.pyc *.py~ build dist

.PHONY: clean
