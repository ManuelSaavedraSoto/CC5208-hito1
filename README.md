# CC5208 - Hito 1

README para el Hito 1 del curso CC5208.

## Situación, Complicación & Propuesta del Hito 0

Situación, complicación y propuestas planteadas para el hito 0.

### Situación

> Existe una pluralidad de series animadas en la actualidad:   
>+ Temporada “Primavera 2025” hay más de 50 series que van a estar en transmisión.
>+ 49 series y múltiples películas confirmadas para “Verano 2025”
>+ Industria del anime es la exportación más grande de Japón.
>+ Se pronostica que la industria solo crecerá, llevando a más series disponibles.

### Complicación

> No está claro qué factores específicos influyen en la popularidad de un anime dentro de estas plataformas.

### Propuesta

> Elaborar un análisis y recopilación de distintas series en diferentes categorías para comparar su impacto en el consumo de anime con el fin de realizar una producción informada a futuro.
> 
>+ Demografía
>+ Género
>+ Temporada
>+ etc.


## Feedback del Hito 0

> Hola,
> 
> Está entretenido este proyecto.
> La situación está muy bien.
> La complicación no tanto. Lo que pasa es que no es una complicación que algunos factores no estén claros. ¿Por qué es un problema? ¿Para quién lo es?
> Su propuesta dice "elaborar un análisis" pero, ¿quién usaría este análisis? Hay muchísimas opciones. Por ejemplo, a Crunchy Roll (por dar un ejemplo) no le importa tanto si el género es relevante o no, mientras que quizás a una productora sí. Entonces, lo que están proponiendo es realmente un análisis exploratorio.
> 
> Para resolver eso les recomiendo lo siguiente:
> En la complicación identifiquen a quién le afecta la complicación, y defínanla de manera específica para esa entidad. Por ejemplo, si es Crunchy Roll, la complicación podría ser que hay series que almacenan espacio y gastan recursos pero no se están viendo. Mientras más crece la base de datos más difícil es navegarla pero no cuesta menos dinero.  
> Y en la propuesta sean más específicos también. Por ejemplo, siguiendo la complicación anterior: medir dependencia entre ranking y múltiples variables, y las tendencias de esta dependencia en el tiempo. Esto aporta valor porque ayudaría a Crunchy Roll a armar un sistema de recomendación.
> 
> Respecto a la factibilidad de la propuesta, el dataset luce bien pero no vimos cuántas series tiene o cuánto abarca el dataset.
> 
> Saludos,
> 
> Edu

## Revisión

Revisión de la complicación y propuesta considerando el feedback recibido.

### Complicación

Para los estudios de animación y productoras de anime, la saturación del mercado presenta varios desafíos críticos:

- La inversión en producción de nuevas series representa un alto riesgo financiero[^1] sin indicadores claros de éxito
- La creciente competencia (50+ series por temporada) dificulta captar y mantener la atención de la audiencia
- Las decisiones creativas y de producción se toman frecuentemente basadas en intuición más que en datos concretos
- La falta de métricas claras sobre preferencias de audiencia puede resultar en pérdidas significativas de inversión

[^1]: Association of Japanese Animation (AJA): [Anime Industry Report 2021](https://aja.gr.jp/english/japan-anime-data) - El reporte destaca los crecientes costos de producción como un desafío significativo para la industria.

### Propuesta

Desarrollar un análisis de datos orientado a optimizar las decisiones de producción para estudios y productoras de anime:

1. Analizar factores críticos de éxito comercial:
   - Relación entre inversión en producción y retorno
   - Impacto del estudio de animación en la recepción
   - Efectividad de diferentes combinaciones de géneros
   - Correlación entre staff clave (directores, guionistas) y éxito

2. Identificar tendencias de mercado:
   - Evolución histórica de géneros populares
   - Patrones de saturación por demografía
   - Temporadas óptimas para diferentes tipos de series
   - Duración ideal de series por género

3. Generar métricas para decisiones de producción:
   - Indicadores de potencial comercial por género y demografía
   - Optimización de recursos de producción
   - Evaluación de riesgo para diferentes formatos de serie

Este análisis permitirá:
- Reducir el riesgo en nuevas producciones
- Optimizar presupuestos de producción
- Identificar nichos de mercado subexplotados
- Mejorar el retorno de inversión

## Datos

Se usarán los csv disponibles en [este](https://www.kaggle.com/datasets/azathoth42/myanimelist) de Kaggle.

Extraer los `.csv` a carpeta `csv/`

## Análisis Preliminar de Datos mediante Copilot

### Evaluación de Datos Disponibles

#### Fuentes de Datos
- **AnimeList.csv/anime_filtered.csv**
  - Información básica de series
  - Métricas de popularidad y ratings
  - Metadata (géneros, estudios, demografía)
  - Fechas de emisión y estado
  
- **UserList.csv/users_filtered.csv**
  - Datos de interacción de usuarios
  - Información demográfica
  - Patrones de consumo
  - Métricas de engagement

#### Viabilidad para Análisis Propuestos

1. **Análisis de Factores de Éxito**
- ✅ Análisis de impacto por estudio
- ✅ Evaluación de combinaciones de géneros
- ❌ Datos de inversión no disponibles
- ❌ Información limitada de staff

2. **Análisis de Tendencias**
- ✅ Evolución temporal de géneros
- ✅ Patrones demográficos
- ✅ Análisis estacional
- ✅ Evaluación de formatos

3. **Métricas de Producción**
- ✅ Indicadores por género/demografía
- ✅ Evaluación de riesgo histórico
- ❌ Datos de recursos de producción

### Visualizaciones Propuestas

1. **Análisis Temporal**
```python
# Series históricas de ratings por género
df.groupby(['year', 'genre'])['rating'].mean()

# Patrones estacionales
df.groupby(['season', 'genre'])['popularity'].aggregate(['mean', 'std'])

# Evolución de saturación de mercado
df.groupby('year')['genre'].value_counts()
```

2. **Métricas de Éxito**
```python
# Rendimiento por estudio
df.groupby('studio')['rating'].describe()

# Distribución por género
df.groupby('genre')['popularity'].agg(['mean', 'median', 'std'])

# Preferencias demográficas
df.pivot_table(values='rating', index='demographic', columns='genre')
```

3. **Análisis de Mercado**
```python
# Correlación género-rating
df.pivot_table(values='rating', index='genre', columns='year')

# Duración óptima por formato
df.groupby(['type', 'episodes'])['rating'].mean()

# Preferencias por región
df.groupby(['region', 'genre'])['popularity'].mean()
```

### Conclusiones Preliminares

1. **Fortalezas del Dataset**
- Cobertura temporal extensa
- Datos demográficos robustos
- Métricas de éxito consistentes

2. **Limitaciones**
- Sin información financiera
- Datos limitados de producción
- Información incompleta de staff

3. **Recomendaciones**
- Enfocar análisis en patrones de audiencia
- Priorizar tendencias de mercado
- Complementar con datos externos de presupuestos

### Próximos Pasos
1. Limpieza y preparación de datos
2. Desarrollo de visualizaciones clave
3. Análisis estadístico detallado
4. Validación de hipótesis iniciales

## Requerimientos

* Python 3

* Pandas