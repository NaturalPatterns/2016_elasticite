default: posts/2015-04-30_trame-sensorielle.ipynb

todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)

# macros for tests
%.pdf: %.ipynb
	ipython nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to latex --post PDF $<

# cleaning macros
touch:
	touch *.ipynb

clean:
	rm -fr  *.tex test_modele_dynamique_files test_coordonees_perceptives_files workflow_files *.dvi *.ps *.out *.log *.aux *.bbl *.blg *.snm *.fls *.nav *.idx *.toc *.fff *.synctex.gz* *.fdb_latexmk

.PHONY:  all clean