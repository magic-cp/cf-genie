# cf-genie

Tesis de grado para optar al título de Ingeniería de la Computación en la Universidad Simón Bolívar

# Instalacion

Como cualquier proyecto de python, primero hay que configurar un ambiente virtual. Luego de crearlo y activarlo, hay que correr el siguiente comando para instalar
las dependencias escenciales:

```shell
pip install -r requirements.txt
```

Si deseas desarrollar, debes insetalar las dependencias de desarrollo:

```shell
pip install -r dev_requirements.txt
```

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
