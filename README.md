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
