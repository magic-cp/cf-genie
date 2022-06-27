# cf-genie

Tesis de grado para optar al título de Ingeniería de la Computación en la Universidad Simón Bolívar

# Instalacion

> La guia de instalacion esta disenada para mi computadora personal, que corre en Apple M1 Max. Algunas cosas pueden variar

Primero, instale [Miniforge3](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh). Luego, execute el siguiente comando para crear un ambiente de _conda_:

```bash
conda  env create --name cf-genie --file environment.yml
```

Este comando va a crear un ambiente de `conda` con todas las dependencias instaladas.


# Uso

Hay varios módulos, y todos se corren de la siguiente manera:

```shell
python -m cf_genie.tasks.<nombre del task>
```

Para ver la lista completa de tasks, ver los archivos dentro de [./cf_genie/tasks](./cf_genie/tasks)

# Setup inicial

Para recrear toda la data para entrenar los modelos, hay que ejecutar los siguientes scripts, en este orden:

1. `load_cf_data`
2. `generate_temp_input_for_raw_dataset`
3. `scrap_dataset`
4. `cleanup_dataset_task`

Entre (3) y (4) hay que mover el archivo `temp/raw_dataset_file.csv` a `dataset/raw_cf_problems.csv`.

# Ejecucion

Usamos `hyperopt` para la hiper-parametrizacion de nuestros modelos. Todas las combinaciones realizadas son guardadas en una base de datos de mongo. Para facilitar el setup del proyecto, se añadió `docker-compose.yml` que permite construir un contenedor de docker con un volumen persistente para guardar los resultados de todas las ejecuciones.

Para levantar el container de docker, solo hay que levantar la red con:

```shell
docker-compose up -d
```

> ℹ️ Sobre `mongo-express`
>
> Incluido en la red esta un contenedor de `mongo-express`, que es un cliente web sencillo para consultar la base de datos. Se puede acceder a este cliente en `localhost:8081`

El mongodb path para conectarse seria `mongo://localhost:27017/admin/jobs`. No hay autenticacion en este esquema (no hace falta!).

Luego, hay que correr el módulo de Python donde se hace la hyper-parametrización con `hyperopt`. Al ejecutarlo, el módulo va a quedarse "idle": lo que ocurre es que se van a generar documentos en MongoDB, donde cada uno tiene la informacion necesaria para generar el modelo (algoritmo de aprendizaje + parametros), pero no va a entrenar ningun modelo.

Para empezar el entrenamiento en paralelo, hay que iniciar los workers. Para iniciar un (y solo un) worker, hay que ejecutar el siguiente comando:

```bash
./hyperopt-worker.sh
```

Si quieres tener N workers, hay que correr este worker N veces en N sesiones de terminal distintas.

Cada worker va a operar de la siguiente manera:

1. Se escoge un documento de MongoDB que no se haya evaluado. Cada documento tiene la información necesaria para poder evaluar la función
2. Cuando el worker termina de entrenar el modelo, va a actualizar el documento con los resultados (usualmente el score y el modelo serializado con `pickle`)

> 🚀 Se puede tener una instancia de MongoDB en la nube, y tener varios workers como maquinas dedicaddas para poder acelerar el proceso. Stonks 📈
