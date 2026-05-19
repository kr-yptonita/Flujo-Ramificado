# Flujo Ramificado (Branched Flow Model)

Este repositorio contiene la implementación y documentación del proyecto de investigación sobre modelado de **Flujos Ramificados (Branched Flows)**. El proyecto abarca desde la simulación de trayectorias estocásticas (como el trazado de fotones a través de una burbuja de jabón) hasta la extensión del modelo para el análisis teórico de ondas sísmicas en el Océano Pacífico.

## Descripción del Proyecto

El flujo ramificado es un fenómeno de propagación de ondas que ocurre en medios débilmente dispersivos. Este repositorio documenta:
1.  **Simulaciones Estocásticas**: El trazado probabilístico de particulas que en este caso serán fotones a través de medios no homogéneos, modelando el espesor y la difracción. Se incluyen algoritmos deterministas y estocásticos con soporte multihilo para un mejor rendimiento.
2.  **Reporte de Investigación**: Todo el trabajo teórico está documentado en un informe de LaTeX accesible en este repositorio.

## Estructura del Repositorio

-   **`Modelo/`**: Contiene todo el código de simulación.
    -   **`Burbuja/`**: Scripts de Python para modelar el espesor de la pared de la burbuja y trazar trayectorias de fotones. Incluye:
        -   `Generador de matrices.py`: Flujo de trabajo de procesamiento de imágenes para generar matrices de espesor basadas en los principios de interferencia de película delgada.
        -   `Trazador_de_fotones.py` y `modelo_grafico_estocastico.py`: Implementan el algoritmo de trazado probabilístico de fotones.
    -   **`Sismo/`**: Directorio de trabajo y código para adaptar el modelo de la burbuja a la simulación de propagación de ondas sísmicas en el Océano Pacífico, tomando en cuenta las barreras físicas.
-   **`LaTeX/`**: Código fuente del reporte de investigación, dividido modularmente por capítulos (`Capitulo_1` al `Capitulo_5`), diagramas en TikZ y bibliografía (`referencias.bib`).
-   **`Articulos/`**: Literatura y artículos de investigación que sirven como base teórica para el proyecto.

## Uso y Ejecución

Para correr las simulaciones, se recomienda usar un entorno virtual de Python. Los scripts se ejecutan desde la carpeta `Modelo/Burbuja`:

```bash
cd Modelo/Burbuja
python modelo_grafico_estocastico.py
```

Para compilar el documento LaTeX:

```bash
cd LaTeX
# Puedes usar pdflatex, latexmk o tu compilador preferido
latexmk -pdf main.tex
```
## Licencia

Este proyecto está distribuido bajo la **Licencia MIT**. Consulta el archivo `LICENSE` para más información.
