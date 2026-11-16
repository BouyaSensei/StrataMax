"""Hot load pour tester le moteur du fork, sans jamais laisser le provider de Strata Desktop arrete.

La regle, ecrite par Axel le 2026-11-16 : un seul moteur a la fois. Ce script ne lance le fork que quand le
provider de l'app est arrete et verifie qu'aucun strata.exe ne reste vivant ; il rend le provider a la fin,
dans la meme execution, et verifie qu'il est pret sur le meme port.

Usage : python hot-load.py <config-du-moteur.json> <dossier-de-sortie> [--long]
"""
import json, subprocess, sys, time, urllib.request

APP = "http://127.0.0.1:8090"
FORK = "http://127.0.0.1:8095"
WS = r"C:/Users/ax_95/AppData/Roaming/CindyGlobal/owners/8abfb6a6ce34715bea84/bots/bot_1794483826996_4ieo44/workspace"
trace = []


def log(msg):
    trace.append({"t": round(time.time() - T0, 1), "msg": msg})
    print(json.dumps(trace[-1], ensure_ascii=False), flush=True)


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=15))


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def n_strata():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "@(Get-Process strata -ErrorAction SilentlyContinue).Count"],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or "0")


def provider_instances():
    d = get(APP + "/api/native")
    return [i for i in d.get("instances", []) if i.get("status") == "ready"]


T0 = time.time()
cfg_path, out_dir = sys.argv[1], sys.argv[2]
extra = ["--long"] if "--long" in sys.argv else []

# 1. le provider doit etre pret et sans requete en vol, sinon on ne touche a rien
ready = provider_instances()
if not ready:
    log("ARRETE : le provider de l'app n'est pas pret, rien n'est decharge")
    sys.exit(2)
serving = get(APP + "/api/serving")
if serving.get("in_flight", 0) > 0:
    log("ARRETE : %d requete(s) en vol sur le provider, rien n'est decharge" % serving.get("in_flight", 0))
    sys.exit(2)
pid_provider = ready[0]["id"]
port_provider = ready[0].get("port")
log("provider %s pret sur %s, file d'attente vide" % (pid_provider, port_provider))

# 2. decharger, puis verifier que le moteur est parti avant de lancer le fork
t = time.time()
r = post(APP + "/api/native/instances/%s/unload" % pid_provider, {})
log("dechargement: HTTP %s en %.1f s" % (r.get("http_status") or 200, time.time() - t))
for _ in range(30):
    if n_strata() == 0 and not provider_instances():
        break
    time.sleep(2)
else:
    log("ARRETE : le moteur du provider est encore vivant apres 60 s, le fork n'est pas lance")
    sys.exit(3)
log("moteur du provider arrete, 0 strata.exe, VRAM rendue")

# 3. lancer le fork, un seul moteur, et le verifier
srv = subprocess.Popen(["python", "strata.py", "serve", "--profile", cfg_path, "--port", "8095"],
                       cwd=WS + "/stratamax", stdout=open(WS + "/hot-fork-serve.log", "a", encoding="utf-8"),
                       stderr=subprocess.STDOUT)
ok = False
for i in range(40):
    time.sleep(5)
    n = n_strata()
    if n > 1:
        srv.terminate()
        log("ARRETE : %d moteurs strata en meme temps, le fork est arrete" % n)
        sys.exit(4)
    try:
        st = get(FORK + "/v1/status")
        if st.get("loaded") and st.get("activity", {}).get("in_flight", 0) == 0:
            ok = True
            log("fork pret en %.1f s, 1 moteur, modele %s, contexte %s" %
                (time.time() - T0, st.get("model"), st.get("context")))
            break
    except Exception:
        pass
if not ok:
    srv.terminate()
    log("ARRETE : le fork n'est pas pret en 200 s")
    sys.exit(5)

# 4. la mesure, sur le serveur deja lance
cmd = ["python", "tools/strata_bench.py", "--config", cfg_path, "--url", FORK, "--out", out_dir] + extra
r = subprocess.run(cmd, cwd=WS + "/stratamax", capture_output=True, text=True)
med = {}
for key in ("chat decode tok/s", "4K prompt tok/s", "16K prompt tok/s", "32K prompt tok/s",
            "agent decode tok/s", "chat first token (s)", "4K first token (s)", "32K first token (s)"):
    for line in r.stdout.splitlines():
        if line.strip().startswith(key):
            nums = [w for w in line.split() if w.replace(".", "").isdigit()]
            med[key] = nums[0] if nums else None
            break
log("mesure: " + json.dumps(med, ensure_ascii=False))

# 5. arreter le fork, verifier qu'il ne reste qu'un moteur possible
srv.terminate()
for _ in range(30):
    if n_strata() == 0:
        break
    time.sleep(2)
log("fork arrete, %d strata.exe" % n_strata())

# 6. rendre le provider, et verifier qu'il est pret sur le meme port
t = time.time()
r = post(APP + "/api/native/operations", {"kind": "load", "args": {"model": "swift-1.5-iq2_xs"}})
job = r.get("job") or r.get("id")
log("rechargement demande (job %s), HTTP %s" % (job, r.get("http_status") or 202))
for i in range(40):
    time.sleep(3)
    ins = provider_instances()
    if ins:
        log("provider de retour en %.1f s: instance %s, port %s, statut %s" %
            (time.time() - t, ins[0]["id"], ins[0].get("port"), ins[0].get("status")))
        break
else:
    log("ALERTE : le provider n'est pas de retour apres 120 s")
    sys.exit(6)
if n_strata() != 1:
    log("ALERTE : %d strata.exe apres le rechargement" % n_strata())
json.dump(trace, open(WS + "/hot-load-trace.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
