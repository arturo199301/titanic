import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Crear el modelo
df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
df['is_top_100'] = (df['rank'] <= 100).astype(int)

features = ['daily_streams', 'daily_stream_share_pct', 'billed_artist_count', 'is_collaboration_int', 'wrapped_global_top10_rank']

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(df[features], df['is_top_100'])

# Caso 1: Canción Solista (1 artista)
cancion_solista = pd.DataFrame([[500000, 0.08, 1, 0, 0]], columns=features)

# Caso 2: Misma canción pero siendo Colaboración (2 artistas)
cancion_colaboracion = pd.DataFrame([[500000, 0.08, 2, 1, 0]], columns=features)

prob_solo = model.predict_proba(cancion_solista)[0][1]
prob_collab = model.predict_proba(cancion_colaboracion)[0][1]

print(f"Probabilidad de ser Top 100 (Solista): {prob_solo * 100:.1f}%")
print(f"Probabilidad de ser Top 100 (Colaboración): {prob_collab * 100:.1f}%")
