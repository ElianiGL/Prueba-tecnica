.PHONY: instalar documentos evaluar prueba-rapida pruebas acuerdo

instalar:
	pip install -r requirements.txt

documentos:
	python -c "from laboral.descarga import descargar; descargar()"

evaluar:
	python -m evaluacion.run --aa

prueba-rapida:
	python -m evaluacion.run --limite 5

pruebas:
	python -m pytest -q

acuerdo:
	python -m evaluacion.acuerdo
