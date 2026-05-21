"""
ask_analyst.py
==============
Stellt Fragen zu den Schneid-Experimentdaten und bekommt Antworten
von einem lokalen Sprachmodell (Ollama) — kein API-Key, kein Internet.

Voraussetzungen:
    1. Ollama installieren:  https://ollama.com  (oder: curl -fsSL https://ollama.com/install.sh | sh)
    2. Modell laden:         ollama pull llama3.1
    3. Dieses Script starten: python ask_analyst.py

Optional: causal_analysis_cutting.py vorher ausführen um aktuelle
Zahlen zu laden. Das Script funktioniert aber auch mit den eingebetteten
Standardzahlen.
"""

import json
import sys
import requests
from pathlib import Path

# =============================================================================
# KONFIGURATION
# =============================================================================

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"          # alternatives: mistral, gemma2, phi3
SUMMARY_PATH = Path("causal_summary.json")   # wird von causal_analysis_cutting.py erzeugt

# =============================================================================
# DATENZUSAMMENFASSUNG (Fallback wenn causal_summary.json nicht vorhanden)
# =============================================================================

DEFAULT_SUMMARY = """
EXPERIMENT: Roboter-Brot-Schneiden (N=1302 Trials)
7 Roboter, 19 Objekte, 5 Seeds. Balanciertes Design: jeder Roboter testete dieselben Objekte.
Gesamterfolgsrate: 39.6%

WICHTIGE KAUSALE BEFUNDE:
- ATE der Perturbation auf Erfolg: -0.683 (IPW, 95% CI [-0.726, -0.638])
- 69% des Effekts läuft durch collision_failure_count (Mediation), 31% direkter Effekt
- perturbation_type (90°/180°) ist ENDOGEN — sequenzieller Recovery-Prozess, kein Treatment
- Objektgröße: stärkster geometrischer Prädiktor (Korrelation -0.307)
- Orientierung (yaw): schwacher Effekt (Korrelation -0.075)

ERFOLGSRATEN NACH ROBOTER:
  rollin_justin : 63.4%  | baseline=80.2%  | perturbed=43.5%  | Koll.Ø=2.11
  pr2           : 58.6%  | baseline=95.4%  | perturbed=27.0%  | Koll.Ø=2.78
  unitree_g1    : 49.5%  | baseline=100%   | perturbed=30.4%  | Koll.Ø=3.75
  tiago         : 43.5%  | baseline=98.1%  | perturbed=21.8%  | Koll.Ø=3.85
  armar7        : 25.3%  | baseline=66.7%  | perturbed=18.2%  | Koll.Ø=4.72
  hsrb          : 20.4%  | baseline=95.2%  | perturbed=10.9%  | Koll.Ø=2.50
  stretch       : 16.7%  | baseline=100%   | perturbed=7.2%   | Koll.Ø=2.59

ERFOLG NACH OBJEKTGRÖSSE:
  XS (≤0.30m): 49.5%  |  S (≤0.34m): 30.1%  |  M (≤0.38m): 17.4%  |  L (>0.38m): 1.3%

ERFOLG NACH ORIENTIERUNG (|yaw|):
  0°: 47.5%  |  45°: 35.2%  |  90°: 36.9%  |  135°+: 39.1%

KOLLISIONSFEHLER vs. ERFOLG:
  0 Fehler: 89.2%  |  1-2 Fehler: ~98% (Recovery gelingt)  |  3 Fehler: 6.2%  |  6 Fehler: 0%

RECOVERY-PROZESS (perturbation_type):
  Primärversuch → scheitert → Rotation +90° → scheitert → Rotation +90° (gesamt 180°) → scheitert → Abbruch
  Nach 1. Rotation (90°):  Erfolg=100% (n=114) — diese haben es noch geschafft
  Nach 2. Rotation (180°): Erfolg=9.5% (n=830) — fast alle scheitern dann

BESTE/SCHLECHTESTE OBJEKTE PRO ROBOTER:
  rollin_justin: best=bread_0017 (100%), worst=bread_0001 (26.7%)
  pr2:           best=bread_0012 (80%),  worst=bread_0009 (20%)
  tiago:         best=bread_0017 (80%),  worst=bread_0019 (0%)
  stretch:       best=bread_0001 (40%),  worst=bread_0005 (0%)
  hsrb:          best=bread_0017 (40%),  worst=bread_0009 (0%)
  armar7:        best=bread_0012 (60%),  worst=bread_0006 (0%)
  unitree_g1:    best=bread_0003 (86.7%), worst=bread_0019 (0%)

ROBOT DECISIONS (gesamt):
  task_failed: 57.7%  |  cut (direkt): 22.0%  |  retry_after_rotation: 14.8%
  retry_with_left_arm: 2.8%  |  skip_object: 2.7%
"""

SYSTEM_PROMPT = """Du bist ein Experte für kausale Inferenz und Robotik-Datenanalyse.
Du hast Zugang zu den vollständigen Ergebnissen eines Roboter-Schneid-Experiments.

{summary}

Regeln:
- Antworte präzise und zitiere konkrete Zahlen aus den Daten.
- Unterscheide klar zwischen kausalen Aussagen (ATE, Mediation) und rein assoziativen Befunden.
- Für kontrafaktische Fragen nutze die Daten für begründete Schätzungen.
- Wenn etwas nicht ableitbar ist, sag das explizit.
- Antworte auf Deutsch wenn auf Deutsch gefragt, auf Englisch wenn auf Englisch.
- Halte Antworten prägnant und informativ."""

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN  = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"

def load_summary():
    if SUMMARY_PATH.exists():
        try:
            data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            # JSON-Summary in lesbaren Text umwandeln
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return DEFAULT_SUMMARY

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            print(f"{YELLOW}Ollama läuft, aber kein Modell geladen.{RESET}")
            print(f"Bitte ausführen: {BOLD}ollama pull {OLLAMA_MODEL}{RESET}")
            return False
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"{YELLOW}Modell '{OLLAMA_MODEL}' nicht gefunden.{RESET}")
            print(f"Verfügbare Modelle: {', '.join(models)}")
            print(f"Bitte ausführen: {BOLD}ollama pull {OLLAMA_MODEL}{RESET}")
            print(f"Oder MODEL in der Konfiguration auf eines der obigen setzen.")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"{RED}Ollama nicht erreichbar.{RESET}")
        print(f"Bitte starten: {BOLD}ollama serve{RESET}  (in separatem Terminal)")
        print(f"Oder installieren: https://ollama.com")
        return False

def ask_ollama(question, history, summary):
    # Konversationsverlauf als Text aufbauen
    history_text = ""
    for h in history[-6:]:   # letzte 3 Runden als Kontext
        role = "Nutzer" if h["role"] == "user" else "Analyst"
        history_text += f"\n{role}: {h['content']}"

    prompt = SYSTEM_PROMPT.format(summary=summary)
    if history_text:
        prompt += f"\n\nBisheriges Gespräch:{history_text}"
    prompt += f"\n\nNutzer: {question}\nAnalyst:"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.3, "num_predict": 600}
    }

    print(f"\n{GREEN}Analyst:{RESET} ", end="", flush=True)
    full_response = ""

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if chunk.get("done"):
                        break
        print("\n")
        return full_response.strip()
    except requests.exceptions.Timeout:
        print(f"\n{YELLOW}Timeout — Modell braucht zu lange. Versuche ein kleineres Modell (phi3, gemma2).{RESET}\n")
        return ""
    except Exception as e:
        print(f"\n{RED}Fehler: {e}{RESET}\n")
        return ""

def print_header():
    print(f"\n{BOLD}{'─'*58}{RESET}")
    print(f"{BOLD}  Cutting Experiment Analyst  (lokal via Ollama){RESET}")
    print(f"{DIM}  Modell: {OLLAMA_MODEL}  |  N=1302 · 7 Roboter · 19 Objekte{RESET}")
    print(f"{BOLD}{'─'*58}{RESET}")
    print(f"{DIM}  Tippe deine Frage und drücke Enter.{RESET}")
    print(f"{DIM}  'q' zum Beenden  |  'reset' für neues Gespräch{RESET}")
    print(f"{BOLD}{'─'*58}{RESET}\n")

BEISPIEL_FRAGEN = [
    "Warum ist Rollin' Justin besser als Stretch?",
    "Ist die Objektgröße kausal für den Misserfolg?",
    "Wenn ich Tiago durch PR2 ersetze, wie ändert sich die Erfolgsrate?",
    "Was sind die wichtigsten Bedingungen für einen erfolgreichen Schnitt?",
    "Welches Objekt ist am schwierigsten zu schneiden?",
]

# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    print_header()

    if not check_ollama():
        sys.exit(1)

    summary = load_summary()
    if SUMMARY_PATH.exists():
        print(f"{GREEN}✓{RESET} Analysezusammenfassung geladen aus {SUMMARY_PATH}\n")
    else:
        print(f"{DIM}  (Eingebettete Standardzusammenfassung wird verwendet.){RESET}")
        print(f"{DIM}  Für aktuellere Zahlen: python causal_analysis_cutting.py{RESET}\n")

    print(f"{DIM}  Beispielfragen:{RESET}")
    for f in BEISPIEL_FRAGEN:
        print(f"  {DIM}· {f}{RESET}")
    print()

    history = []

    while True:
        try:
            question = input(f"{CYAN}Du:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Tschüss.{RESET}\n")
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit", "tschüss", "bye"):
            print(f"\n{DIM}Tschüss.{RESET}\n")
            break
        if question.lower() == "reset":
            history = []
            print(f"{DIM}  Gespräch zurückgesetzt.{RESET}\n")
            continue
        if question.lower() == "modell":
            print(f"{DIM}  Aktuelles Modell: {OLLAMA_MODEL}{RESET}\n")
            continue

        answer = ask_ollama(question, history, summary)
        if answer:
            history.append({"role": "user",    "content": question})
            history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
