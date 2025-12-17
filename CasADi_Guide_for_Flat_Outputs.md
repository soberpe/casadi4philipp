# CasADi: Technische Referenz

Alternative zu Mathematica für rekursive Jacobi-Berechnungen. Löst das Problem der Ausdrucksexplosion durch Verwendung von Berechnungsgraphen statt symbolischer Ausdrücke.

**Kernvorteil:** Algorithmische Differentiation verhindert Explosion - egal wie komplex oder rekursiv die Ableitungen werden.

---

## Installation

```bash
pip install casadi numpy matplotlib
```

---

## Grundkonzepte

### 1. Symbolische Variablen

```python
import casadi as ca

# Skalare symbolische Variablen
x = ca.SX.sym('x')
y = ca.SX.sym('y')

# Vektor symbolische Variablen
x = ca.SX.sym('x', 5)      # 5-dimensionaler Vektor
X = ca.SX.sym('X', 3, 3)   # 3x3 Matrix
```

**Wann verwenden:**
- `SX` (Scalar eXpressions): Kleine-mittlere Probleme, sparse Jacobi-Matrizen
- `MX` (Matrix eXpressions): Sehr große Probleme, dichte Operationen

### 2. Funktionen erstellen

```python
# Symbolischen Ausdruck definieren
f = ca.sin(x[0])**2 + x[1] * ca.exp(x[2])

# Auswertbare Funktion erstellen
f_func = ca.Function('f', [x], [f])

# Numerisch auswerten
result = f_func([1.0, 2.0, 0.5])
```

### 3. Automatische Differentiation

```python
# Jacobi-Matrix automatisch berechnen
J = ca.jacobian(f, x)

# Funktion erstellen, die f und J zurückgibt
f_and_J = ca.Function('f_J', [x], [f, J])

# Beides auf einmal auswerten
f_val, J_val = f_and_J([1.0, 2.0, 0.5])
```

---

## Beispiel 1: Jacobi-Matrix berechnen

Berechnung der Jacobi-Matrix einer Vektorfunktion $f: \mathbb{R}^n \to \mathbb{R}^m$:

```python
import casadi as ca
import numpy as np

n = 10  # Eingangsdimension
m = 8   # Ausgangsdimension

x = ca.SX.sym('x', n)

# Vektorfunktion definieren
f = ca.vertcat(
    ca.sin(x[0]) * x[1]**2 + ca.exp(x[2]),
    x[0]**3 - x[1]*x[2] + x[3],
    ca.norm_2(x[0:4]),
    ca.sumsqr(x[4:8])
)

# Jacobi-Matrix df/dx (m x n Matrix) berechnen
J = ca.jacobian(f, x)

# Auswertbare Funktion erstellen
f_func = ca.Function('f', [x], [f, J])

# Auswerten
x_val = np.random.randn(n)
f_val, J_val = f_func(x_val)
```

**Wichtig:** Egal wie komplex `f` ist, CasADi speichert es als Graph. Die Jacobi-Berechnung ist effizient und das Ergebnis bleibt handhabbar.

---

## Beispiel 2: Rekursive Jacobi-Vektor-Produkte

Für deine Anwendung benötigst du oft Ableitungen von Ableitungen. So berechnest du:
- $J \cdot v$ (Richtungsableitung)
- $\frac{\partial(J \cdot v)}{\partial x} \cdot v$ (zweite Ordnung)
- $\frac{\partial^2(J \cdot v)}{\partial x^2} \cdot v$ (dritte Ordnung)

```python
x = ca.SX.sym('x', 5)
v = ca.SX.sym('v', 5)  # Richtungsvektor

# Funktion definieren
f = ca.vertcat(
    x[0]**2 * ca.sin(x[1]) + ca.exp(x[2]*x[3]),
    x[0] * x[1] * x[2] - x[3]**3 + x[4]
)

# Erste Ordnung: J * v
J = ca.jacobian(f, x)
Jv = ca.mtimes(J, v)

# Zweite Ordnung: Ableitung von (J*v) nach x, mal v
Jv_x = ca.jacobian(Jv, x)
Jvv = ca.mtimes(Jv_x, v)

# Dritte Ordnung: Rekursion fortsetzen
Jvv_x = ca.jacobian(Jvv, x)
Jvvv = ca.mtimes(Jvv_x, v)

# Funktion für alle Ableitungen erstellen
derivatives_func = ca.Function('derivatives', 
                              [x, v], 
                              [f, Jv, Jvv, Jvvv])
```

**Wichtig:** Diese rekursive Differentiation würde in Mathematica massive Ausdrücke erzeugen. In CasADi wird es effizient über den Berechnungsgraphen behandelt!

---

## Beispiel 3: Effiziente Forward/Reverse Modes

Für große Probleme ist die Berechnung der vollen Jacobi-Matrix verschwenderisch, wenn du nur $J \cdot v$ oder $v^T \cdot J$ brauchst.

### Forward Mode (für $J \cdot v$)

```python
n = 100  # Große Eingangsdimension
m = 50   # Ausgangsdimension

x = ca.SX.sym('x', n)
v = ca.SX.sym('v', n)

f = # ... irgendeine komplexe Vektorfunktion ...

# Effizient: J*v direkt berechnen ohne volle J zu bilden
Jv = ca.jtimes(f, x, v)  # Forward mode
```

**Komplexität:** 
- Volle Jacobi-Matrix: $O(m \cdot n)$
- Forward mode: $O(m)$ (unabhängig von $n$!)

### Reverse Mode (für $v^T \cdot J$)

```python
# Effizient: v^T * J berechnen ohne volle J zu bilden
vTJ = ca.jtimes(f, x, v, True)  # Reverse mode (transpose=True)
```

**Faustregel:**
- $m \ll n$: Reverse mode verwenden
- $m \gg n$: Forward mode verwenden
- Volle Jacobi-Matrix nötig: `ca.jacobian(f, x)` verwenden

---

## Beispiel 4: Newton-Solver mit automatischer Differentiation

Das mächtigste Feature: **Solver berechnen Jacobi-Matrizen automatisch!**

```python
x = ca.SX.sym('x', 5)

# Gleichungssystem F(x) = 0 definieren
F = ca.vertcat(
    x[0]**2 + x[1]**2 - 4,
    x[2] - ca.sin(x[0]) - ca.cos(x[1]),
    x[3]**3 - x[0]*x[2] + 1,
    x[4] - ca.exp(-x[3]) + 0.5,
    ca.sum1(x) - 2
)

# Newton-Solver erstellen
# Die Jacobi-Matrix dF/dx wird AUTOMATISCH berechnet!
newton_solver = ca.rootfinder('newton_solver', 'newton', 
                             {'x': x, 'g': F})

# Lösen
x0 = np.array([1.0, 1.0, 0.5, 0.5, -1.0])
solution = newton_solver(x0, [])
```

**Du musst die Jacobi-Matrix nie manuell berechnen oder kodieren.** CasADi behandelt das intern mit algorithmischer Differentiation.

---

## Funktionsreferenz

### Differentiation

| Funktion | Zweck | Wann verwenden |
|----------|-------|----------------|
| `ca.jacobian(f, x)` | Volle Jacobi-Matrix $\frac{\partial f}{\partial x}$ | Volle Matrix benötigt |
| `ca.jtimes(f, x, v)` | Forward mode $J \cdot v$ | $m \gg n$ oder Richtungsableitung |
| `ca.jtimes(f, x, v, True)` | Reverse mode $v^T \cdot J$ | $m \ll n$ |
| `ca.gradient(f, x)` | Gradient $\nabla f$ (für skalares $f$) | Optimierung |
| `ca.hessian(f, x)` | Hesse-Matrix $\nabla^2 f$ | Methoden zweiter Ordnung |

### Solver

| Funktion | Zweck | Auto-Diff |
|----------|-------|-----------|
| `ca.rootfinder()` | Löse $F(x) = 0$ | ✅ Jacobi-Matrix |
| `ca.nlpsol()` | Nichtlineare Optimierung | ✅ Gradient & Hesse-Matrix |

### Hilfsfunktionen

```python
ca.vertcat(a, b, c)     # Vektoren vertikal stapeln
ca.horzcat(a, b, c)     # Vektoren horizontal stapeln
ca.mtimes(A, B)         # Matrix-Multiplikation
ca.sum1(x)              # Alle Elemente summieren
ca.sumsqr(x)            # Quadratsumme
ca.norm_2(x)            # Euklidische Norm
```

---

## Performance-Tipps

### 1. Den richtigen Ausdruckstyp wählen

- **SX (Scalar):** Besser für sparse Probleme, viele Operationen, automatische Vereinfachung
- **MX (Matrix):** Besser für sehr große dichte Probleme

```python
# Für die meisten Anwendungen
x = ca.SX.sym('x', n)  # SX verwenden

# Nur bei großen dichten Matrizen
X = ca.MX.sym('X', 100, 100)  # MX verwenden
```

### 2. Forward/Reverse Mode effizient nutzen

```python
# SCHLECHT: Volle Jacobi-Matrix berechnen, wenn nur J*v benötigt
J = ca.jacobian(f, x)  # O(m*n)
Jv = ca.mtimes(J, v)

# GUT: Forward mode direkt verwenden
Jv = ca.jtimes(f, x, v)  # O(m)
```

### 3. Code-Generierung für Produktion

```python
# Effizienten C-Code generieren
f_func.generate('my_function.c')

# Oder JIT-Compilation verwenden (benötigt Compiler)
opts = {'jit': True, 'compiler': 'gcc'}
f_func_fast = ca.Function('f', [x], [f], opts)
```

### 4. Sparsity ausnutzen

CasADi erkennt und nutzt Sparsity automatisch. Um zu helfen:

```python
# CasADi verfolgt Sparsity automatisch
J = ca.jacobian(f, x)

# Sparsity-Pattern prüfen
sparsity = J.sparsity()
print(f"Nonzeros: {sparsity.nnz()}/{sparsity.numel()}")
```

---

## Häufige Patterns

### Pattern 1: Mehrere Ableitungen in einer Funktion

```python
# Funktion erstellen, die f, Jacobi-Matrix und Hesse-Matrix zurückgibt
x = ca.SX.sym('x', n)
f = # ... skalare Funktion ...

grad_f = ca.gradient(f, x)
hess_f = ca.hessian(f, x)[0]

func = ca.Function('f_grad_hess', [x], [f, grad_f, hess_f])
```

### Pattern 2: Parametrische Funktionen

```python
# Funktion mit Parametern
x = ca.SX.sym('x', n)
p = ca.SX.sym('p', m)  # Parameter

f = # ... Funktion von x und p ...

func = ca.Function('f', [x, p], [f])

# Mit spezifischen Parametern auswerten
result = func(x_val, p_val)
```

### Pattern 3: Zeitvariante Trajektorien

```python
# Zustand entlang Trajektorie auswerten
t_values = np.linspace(0, T, 100)
states = []

for t in t_values:
    flat_val = compute_flat_outputs(t)  # Deine Trajektorie
    state_val, _ = state_func(flat_val)
    states.append(state_val)
```

---

## Ressourcen und Dokumentation

- **Offizielle Website:** [https://web.casadi.org/](https://web.casadi.org/)
- **Dokumentation:** [https://web.casadi.org/docs/](https://web.casadi.org/docs/)
- **Forum:** [https://groups.google.com/g/casadi-users](https://groups.google.com/g/casadi-users)
- **GitHub:** [https://github.com/casadi/casadi](https://github.com/casadi/casadi)

### Empfohlene Literatur

1. **CasADi Paper:** Andersson et al., "CasADi – A software framework for nonlinear optimization and optimal control" (2019)
2. **Tutorial:** Verfügbar unter [https://web.casadi.org/get-started/](https://web.casadi.org/get-started/)
3. **Beispiele:** Durchsuche Beispiele im GitHub Repository
