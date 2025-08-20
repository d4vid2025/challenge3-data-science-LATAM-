
# Informe de Análisis y Predicción de Churn de Clientes

## 1. Introducción
El objetivo de este análisis es comprender y predecir la **evasión de clientes** (churn) en una empresa de telecomunicaciones, utilizando técnicas de **machine learning**. Se busca identificar los factores que influyen en la cancelación de servicios y construir un modelo predictivo que permita anticipar esta conducta.

---

## 2. Descripción del Dataset
El dataset contiene **7,032 registros** y **20 variables** después de la limpieza y transformación. Las variables incluyen características demográficas, de servicios contratados y de cuentas:

- Variables demográficas: `customer_gender`, `customer_seniorcitizen`, `customer_partner`, `customer_dependents`.
- Variables de servicios: `phone_phoneservice`, `phone_multiplelines`, `internet_onlinesecurity`, `internet_onlinebackup`, `internet_deviceprotection`, `internet_techsupport`, `internet_streamingtv`, `internet_streamingmovies`.
- Variables de cuenta: `account_contract`, `account_paperlessbilling`, `account_paymentmethod`, `account_charges_monthly`, `account_charges_total`, `cuenta_diarias`.
- Variable objetivo: `churn` (0 = No canceló, 1 = Canceló).

Se realizaron las siguientes transformaciones:

- Codificación de variables categóricas `account_contract` y `account_paymentmethod`.
- Conversión de `account_charges_total` a tipo `float64`.
- Eliminación de la variable `internet_internetservice`.
- Aplicación de SMOTE para balancear la variable objetivo.

---

## 3. Análisis Exploratorio

### 3.1 Proporción de Churn
El dataset original presenta un desbalance:

| Churn | Cantidad |
|-------|----------|
| 0     | 5163     |
| 1     | 1869     |

Se aplicó **SMOTE** para equilibrar las clases, generando **10,326 registros**.

### 3.2 Estadísticas Descriptivas
Algunas métricas relevantes:

- Promedio de gasto mensual: 64.8
- Promedio de gasto total: 2,283.3
- Promedio de cuenta diaria: 2.16
- Edad promedio de los clientes: 32.4 meses de antigüedad (tenure)

### 3.3 Correlación
Las variables más correlacionadas con el churn fueron:

| Variable                     | Correlación |
|-------------------------------|------------|
| cuenta_diarias                | 0.193      |
| account_charges_monthly       | 0.193      |
| account_paperlessbilling      | 0.191      |
| customer_seniorcitizen        | 0.151      |
| customer_tenure               | -0.354     |
| account_contract              | -0.396     |
| internet_onlinesecurity       | -0.332     |
| internet_techsupport          | -0.329     |

Se observa que los clientes con contratos más largos (`Two year`) y mejor soporte técnico tienden a cancelar menos.

### 3.4 Visualización
#### Gasto total vs Churn
![Stripplot gasto total](ruta_a_grafico.png)

#### Tipo de contrato vs Churn
![Stripplot contrato](ruta_a_grafico.png)

---

## 4. Preparación de Datos
- Separación en variables predictoras `X` y variable objetivo `y`.
- División en **train/test** (70/30).
- Normalización de datos con `StandardScaler`.
- Aplicación de SMOTE para balancear clases.

---

## 5. Modelado
Se entrenó un **modelo de regresión logística**:

```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
````

### 5.1 Resultados de Evaluación

| Clase        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.84      | 0.74   | 0.79     | 1593    |
| 1            | 0.75      | 0.85   | 0.80     | 1505    |
| **Accuracy** | 0.79      | -      | -        | 3098    |

El modelo logra un **accuracy del 79%**, con un buen balance entre precisión y recall para ambas clases.

---

## 6. Conclusiones

* Las variables más influyentes para predecir churn son: tipo de contrato, antigüedad del cliente, soporte técnico y pagos electrónicos.
* Los clientes con contratos a largo plazo y con soporte técnico tienden a cancelar menos.
* La aplicación de SMOTE permitió balancear las clases y mejorar la capacidad predictiva del modelo.
* La regresión logística logra una precisión y recall equilibrados, siendo útil para predecir la evasión de clientes y tomar decisiones estratégicas.

---

## 7. Próximos pasos

* Probar modelos más complejos como **Random Forest** o **XGBoost** para mejorar el rendimiento.
* Realizar **feature engineering** con interacciones entre variables.
* Implementar un **dashboard de seguimiento** de churn para visualización en tiempo real.

```

---

```

