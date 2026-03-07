;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "" "12pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("wrapfig" "") ("fontenc" "T2A") ("inputenc" "utf8") ("babel" "russian") ("geometry" "" "top=25mm" "bottom=25mm" "left=30mm" "right=20mm" "top=20mm" "left=3mm" "left=20mm") ("amsmath" "") ("amssymb" "") ("graphicx" "") ("booktabs" "") ("array" "") ("caption" "") ("subcaption" "") ("float" "") ("hyperref" "") ("xcolor" "") ("enumitem" "") ("microtype" "") ("indentfirst" "") ("setspace" "") ("parskip" "")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "fontenc"
    "inputenc"
    "babel"
    "geometry"
    "amsmath"
    "amssymb"
    "graphicx"
    "booktabs"
    "array"
    "caption"
    "subcaption"
    "float"
    "hyperref"
    "xcolor"
    "enumitem"
    "microtype"
    "indentfirst"
    "setspace"
    "parskip")
   (LaTeX-add-labels
    "eq:f1macro"
    "eq:als"
    "eq:snv"
    "tab:bands"
    "tab:loao"
    "tab:ensemble"
    "tab:clf_report"))
 :latex)

