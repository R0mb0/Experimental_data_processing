# LSM

```
Questo progetto riguarda la stima dei parametri di una funzione sulla base di misure sperimentali accoppiate indipendenti.

Le misure sperimetali sono riportate nel file di dati qui sotto. Consistono in 9 misure accoppiate (x, y_mean, std_y), dove la grandezza x è supposta conosciuta con grande precisione mentre la gradezza y è il risultato della media di dieci misurazioni per ogni valore della x; viene pertanto riportato anche il valore della std della media y.

Si suppone che la y è legata alla x da una forma funzionale. In questo esercizio le forme funzionali che supponiamo sono tre:

y = t0 * x^t1

y = t0 + t1*x + t2*x^2

y = t0 + t1*x + t2*exp(x)

Scopo dell'esercizio è stimare i parametri t0, t1, t2 col metodo LS per le tre funzioni e utilizzare il test di pearson col chi2 per decidere quale delle tre funzioni rappresenta meglio l'evoluzione della y come funzione della x.

Il metodo LS si dovrà applicare, quando possibile, nella sua forma analitica o altrimenti utilizzando una minimizzazione numerica (per esempio con scipy.optimize.curve_fit() ); in quest'ultimo caso il chi2 verrà calcolato utilizzando la sua definizione
```
