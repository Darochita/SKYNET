
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semilla_S1_CA.py — "Semilla S1": Autómata Celular continuo con regla mínima + búsqueda QD
- Regla local compartida (3x3) que mezcla canales: [masa, señal, memoria, modulador]
- Actualización ASINCRÓNICA: cada paso se actualiza solo una fracción p de celdas (aleatorio)
- Sin "premios" en la regla: la evaluación usa VIABILIDADES (umbrales) y luego QD (llenar mapa)
- Entrada: un "poke" local (pulso en modulador) a mitad de la simulación
- Descriptores para QD: densidad final (ρ), respuesta al poke (Δ), oscilación (σ temporal)
- Viabilidades: no-extinción, no-explosión, estabilidad suave y respuesta mínima

Ejecuta:
  python semilla_S1_CA.py
Requisitos: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
from dataclasses import dataclass

# Backend opcional en GPU con PyTorch (conv2d + padding circular)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    TORCH_HAS_CUDA = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False
    TORCH_HAS_CUDA = False

# ------------------------------
# Núcleo de regla (compartida)
# ------------------------------
def tanh(x): return np.tanh(x)
def sigmoid(x): return 1/(1+np.exp(-x))

@dataclass
class Config:
    # Tamaño de la grilla (alto x ancho)
    alto:int=36                 # píxeles en eje Y
    ancho:int=36                # píxeles en eje X
    # Canales del estado
    canales:int=4               # [0: masa (0..1), 1: señal, 2: memoria, 3: modulador]
    # Dinámica de actualización
    paso:float=0.1              # escala de integración del delta en cada paso
    frac_async:float=0.6        # fracción de celdas actualizadas por paso (asincronía)
    # Inicialización de genoma
    escala_init:float=0.2       # desviación estándar de pesos y sesgos iniciales
    # Simulación por evaluación
    pasos:int=200               # pasos simulados para calcular descriptores/viabilidades
    semillas:int=1              # cantidad de “semillas” de masa inicial
    # Mapa QD (ejes densidad y respuesta)
    bins:int=10                 # resolución de binning para densidad y respuesta Δ
    iteraciones:int=500         # iteraciones totales del bucle QD
    # Reproducibilidad / backend
    semilla_global:int=7        # semilla base para RNG
    backend:str="torch"         # "auto" | "numpy" | "torch" (GPU si disponible)
    # Estímulo y medición de respuesta
    poke_intensidad:float=2.5   # intensidad del “poke” circular al canal modulador
    poke_radio:int=3            # radio del disco de estímulo
    resp_ventana:int=5          # ventana (en pasos) tras el poke para medir Δ promedio
    # Umbrales de viabilidad (filtros duros)
    th_no_extincion_min:float=0.10  # fracción mínima viva al final
    th_no_explosion_max:float=0.70  # fracción máxima viva permitida (evitar saturación)
    th_oscil_max:float=0.06         # varianza temporal máxima de la densidad (estabilidad)
    th_resp_min:float=0.02          # respuesta mínima Δ tras el poke
    # Referencia de respuesta para binning del mapa QD
    resp_ref:float=0.06         # escala para convertir Δ a bins (eje Y del mapa)
    # Evaluación paralela (ruta torch batch)
    batch:int=1                 # tamaño de lote (B>1 usa eval batch en torch sin cómputo)
    # Logging
    log_every:int=10            # frecuencia de logs en iteraciones QD
    # Asincronía: refresco de la máscara
    async_refresh_every:int=1   # 0=constante; K>0 refresca cada K pasos
    # Momento del poke (0: mitad de la simulación)
    warmup:int=20               # paso en el que se aplica el poke
    # Robustez (sólo ruta no-batch): promediar sobre varias semillas
    avg_seeds:int=1
    # Eje de cómputo (reservorio ligero)
    bins_comp:int=6             # resolución del eje de cómputo (tercer eje del mapa)
    comp_T:int=60               # longitud de la secuencia para test de cómputo
    comp_embed:int=12           # embedding temporal (ventana) para el readout
    comp_quad:bool=True         # añadir términos cuadráticos a las features
    comp_acc_thr:float=0.7      # umbral para considerar “buen cómputo” (reportes)
    comp_amp:float=1.6          # amplitud del estímulo por bit=1 en el test de cómputo
    comp_radius:int=2           # radio del estímulo por bit en el test de cómputo
    # Priorización de cómputo en reemplazos del mapa QD
    comp_priority:bool=True     # si True, mejoras claras en comp desplazan élites
    comp_priority_eps:float=0.03# margen mínimo de mejora en comp para reemplazar
    # Multitarea/robustez del test de cómputo
    comp_k_list: tuple = (3,5)  # tamaños k de paridad a evaluar y promediar
    comp_cv_folds:int=3         # # de folds contiguos para validación cruzada
    comp_seeds:int=2            # repeticiones con distintos streams de bits

# === Métricas de complejidad (espacio) ===
def entropia_01(img, bins=16):
    h = np.clip(img, 0.0, 1.0)
    counts, _ = np.histogram(h, bins=bins, range=(0,1), density=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(bins))

def densidad_bordes(img):
    gy = np.abs(np.roll(img,-1,0) - np.roll(img,1,0))
    gx = np.abs(np.roll(img,-1,1) - np.roll(img,1,1))
    g = 0.5*(gx+gy)
    return float(np.mean(g > 0.1))

# ------------------------------
# Test de cómputo ligero (reservorio)
# ------------------------------
def _score_reservorio_ligero(pesos, sesgo, cfg:Config, rng) -> float:
    """Evalúa capacidad de cómputo con multitarea de paridad k∈comp_k_list
    usando un readout lineal (ridge) con embedding temporal y z-score.
    Robustece con validación cruzada por bloques y promedio sobre comp_seeds.
    Retorna acc∈[0,1].
    """
    try:
        # Construye un S1 numpy (más liviano) sin cambiar cfg global
        s1 = S1(cfg, pesos=pesos, sesgo=sesgo, rng=rng)
        est = s1.init_estado()
        H,W,C = est.shape
        cx, cy = H//2, W//2
        # anillo pequeño derivado del tamaño
        r_in = max(2, min(H,W)//10)
        r_out = r_in + 3
        Y, X = np.ogrid[:H, :W]
        R2 = (Y-cx)**2 + (X-cy)**2
        ring = (R2 >= r_in*r_in) & (R2 <= r_out*r_out)
        idx = np.where(ring.ravel())[0]

        # parámetros reforzados (defaults pueden ser sobreescritos por CLI)
        T = int(max(cfg.comp_T, 100))

        def ridge_acc(Ftr, ytr, Fte, yte):
            lam = 5e-2
            A = Ftr.T @ Ftr + lam*np.eye(Ftr.shape[1])
            b = Ftr.T @ ytr
            try:
                w = np.linalg.solve(A,b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
            yhat = (Fte @ w) > 0.5
            return float((yhat == yte).mean())

        def one_seed_accuracy() -> float:
            # RNG local para stream de bits y posiciones del anillo
            if hasattr(rng, 'integers'):
                rloc = np.random.default_rng(int(rng.integers(0, 2**31-1)))
            else:
                rloc = np.random.default_rng()
            bits = rloc.integers(0,2,size=T)
            # simulación con inyección distribuida sobre el anillo
            # reiniciar estado/máscara para evitar acople entre seeds
            est_loc = est.copy()
            s1.mask_async = None; s1._step = 0
            X_feats = []
            for t in range(T):
                if bits[t] == 1:
                    flat = int(rloc.integers(0, len(idx)))
                    iy, ix = np.unravel_index(idx[flat], (H, W))
                    est_loc = s1.paso(est_loc, estimulo=(iy, ix, float(cfg.comp_amp), int(cfg.comp_radius)))
                else:
                    est_loc = s1.paso(est_loc, estimulo=None)
                X_feats.append(est_loc[...,0].ravel()[idx])
            Xmat = np.stack(X_feats, axis=0)
            # embedding temporal
            E = int(max(cfg.comp_embed, 8))
            feats = []
            for t in range(T):
                start = max(0, t-E+1)
                pad = E - (t - start + 1)
                block = Xmat[start:t+1]
                if pad>0:
                    block = np.pad(block, ((pad,0),(0,0)))
                feats.append(block.reshape(-1))
            F = np.stack(feats, axis=0)
            # Normalización (z-score)
            F_mean = F.mean(axis=0, keepdims=True)
            F_std  = F.std(axis=0, keepdims=True) + 1e-6
            Fz = (F - F_mean) / F_std
            if cfg.comp_quad:
                Fz = np.concatenate([Fz, Fz*Fz], axis=1)

            # Folds contiguos
            folds = max(2, int(getattr(cfg, 'comp_cv_folds', 3)))
            sizes = [T//folds + (1 if r < (T % folds) else 0) for r in range(folds)]
            idxs = []
            a = 0
            for s in sizes:
                idxs.append((a, a+s))
                a += s

            accs_k = []
            k_list = getattr(cfg, 'comp_k_list', (3,5))
            for k in k_list:
                # etiquetas de paridad
                y = np.zeros(T, dtype=np.float64)
                for t in range(T):
                    if t >= k-1:
                        y[t] = int(np.sum(bits[t-k+1:t+1]) % 2)
                # CV
                fold_accs = []
                for (a,b) in idxs:
                    te = np.arange(a,b)
                    tr_l = []
                    if a>0:
                        tr_l.append(np.arange(0,a))
                    if b<T:
                        tr_l.append(np.arange(b,T))
                    tr = np.concatenate(tr_l) if tr_l else np.array([], dtype=int)
                    if tr.size < 10 or te.size < 5:
                        continue
                    fold_accs.append(ridge_acc(Fz[tr], y[tr], Fz[te], y[te]))
                if fold_accs:
                    accs_k.append(float(np.mean(fold_accs)))
            return float(np.mean(accs_k)) if accs_k else 0.0

        S = max(1, int(getattr(cfg, 'comp_seeds', 1)))
        accs = [one_seed_accuracy() for _ in range(S)]
        return float(np.clip(np.mean(accs), 0.0, 1.0))
    except Exception:
        return 0.0

# Descriptor auxiliar (opcional): velocidad de frente post-poke
def velocidad_frente(hist: np.ndarray, t0: int, ventana: int) -> float:
    """Velocidad media del frente tras el poke: promedio temporal del cambio
    paso a paso |hist[t]-hist[t-1]| en la ventana post-poke. Normalizado en [0,1]."""
    T = hist.shape[0]
    w = max(1, int(ventana))
    t_ini = t0 + 1
    t_fin = min(T - 1, t0 + w)
    if t_ini > t_fin:
        return 0.0
    deltas = [float(np.mean(np.abs(hist[t] - hist[t-1]))) for t in range(t_ini, t_fin + 1)]
    if not deltas:
        return 0.0
    return float(np.mean(deltas))

# Utilidad: animación MEJORADA del canal 'masa' usando Matplotlib
def render_anim(hist: np.ndarray, out_path: str = "S1_anim.gif", fps: int = 12,
                cfg=None, desc=None):
    """
    Renderiza animación mejorada del autómata celular con:
    - Colormap más informativo (viridis)
    - Indicadores temporales y de poke
    - Información de descriptores
    - Barra de progreso temporal
    """
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as patches
    
    T = hist.shape[0]
    
    # Determinar momento del poke
    if cfg and cfg.warmup > 0:
        t_poke = cfg.warmup
    else:
        t_poke = T // 2
    
    # Configuración de figura mejorada
    fig, (ax_main, ax_progress) = plt.subplots(2, 1, figsize=(7, 7),
                                               gridspec_kw={'height_ratios': [5, 2]})
    
    # Panel principal: autómata
    # Usar el mismo colormap que en S1_evolucion para coherencia visual
    im = ax_main.imshow(hist[0], cmap='gray', vmin=0, vmax=1, animated=True)
    ax_main.set_title('Autómata Celular S1 - Evolución Temporal', fontsize=12, pad=8)
    
    # Información de descriptores si está disponible
    if desc is not None:
        info_text = f'ρ={desc[0]:.3f}, Δ={desc[1]:.3f}, σ={desc[2]:.4f}, comp={desc[3]:.3f}'
        ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes, 
                    fontsize=10, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_main, shrink=0.8)
    cbar.set_label('Masa', fontsize=10)
    
    # Panel inferior: gráfica de densidad temporal + marcador animado
    dens = hist.mean(axis=(1, 2))
    ax_progress.set_xlim(0, T - 1)
    ymin, ymax = float(dens.min()) - 0.02, float(dens.max()) + 0.02
    if ymin == ymax:
        ymax = ymin + 1.0
    ax_progress.set_ylim(ymin, ymax)
    base_line, = ax_progress.plot(np.arange(T), dens, color='tab:blue', lw=1.8, alpha=0.85)
    marker, = ax_progress.plot([0], [dens[0]], marker='o', color='tab:orange', ms=6)
    ax_progress.axvline(t_poke, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax_progress.text(t_poke, ymax, f'Poke (t={t_poke})', ha='center', va='top',
                     fontsize=9, color='red')
    ax_progress.set_xlabel('Tiempo (pasos)', fontsize=10)
    ax_progress.set_ylabel('Densidad media', fontsize=10)
    ax_progress.set_title('Evolución de la densidad', fontsize=10)
    ax_progress.grid(True, alpha=0.25)

    # Elementos animados adicionales sobre el panel superior
    time_text = ax_main.text(0.98, 0.02, '', transform=ax_main.transAxes,
                             fontsize=11, va='bottom', ha='right', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    phase_text = ax_main.text(0.02, 0.02, '', transform=ax_main.transAxes,
                              fontsize=10, va='bottom', ha='left', fontweight='bold')
    
    def update(t):
        # Actualizar imagen principal
        im.set_data(hist[t])
        
        # Actualizar indicadores temporales
        time_text.set_text(f't = {t:3d}/{T-1}')
        # Mover marcador en la curva de densidad
        marker.set_data([t], [dens[t]])

        # Indicador de fase
        if t < t_poke:
            phase_text.set_text('Pre-poke')
            phase_text.set_color('blue')
        elif t == t_poke:
            phase_text.set_text('¡POKE!')
            phase_text.set_color('red')
        else:
            phase_text.set_text('Post-poke')
            phase_text.set_color('green')
        
        return [im, marker, time_text, phase_text]
    
    # Crear animación
    anim = FuncAnimation(fig, update, frames=range(T), interval=int(1000 / max(1, fps)),
                         blit=True, repeat=True)
    
    plt.tight_layout()
    
    saved = False
    # Intento 1: MP4 (preferido)
    try:
        mp4_path = out_path if out_path.lower().endswith('.mp4') else out_path.rsplit('.',1)[0] + '.mp4'
        anim.save(mp4_path, fps=fps, bitrate=1800)
        saved = True
        print(f"Animación mejorada guardada: {mp4_path}")
    except Exception as e:
        print(f"No se pudo guardar MP4: {e}")
    
    # Intento 2: GIF con PillowWriter (fallback)
    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            gif_path = out_path if out_path.lower().endswith('.gif') else out_path.rsplit('.',1)[0] + '.gif'
            anim.save(gif_path, writer=PillowWriter(fps=fps))
            saved = True
            print(f"Animación guardada como GIF: {gif_path}")
        except Exception as e:
            print(f"No se pudo guardar GIF: {e}")
    
    # Intento 3: volcado a frames PNG
    if not saved:
        frames_dir = os.path.splitext(out_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for t in range(T):
            plt.figure(figsize=(4,4))
            plt.imshow(hist[t], cmap='viridis', vmin=0, vmax=1)
            plt.title(f'S1 - Frame {t:04d}/{T-1} {"(POKE)" if t==t_poke else ""}')
            plt.colorbar(label='Masa')
            plt.savefig(os.path.join(frames_dir, f"frame_{t:04d}.png"), dpi=100, bbox_inches='tight')
            plt.close()
        print(f"Frames guardados en: {frames_dir}")
    
    plt.close(fig)

class S1:
    """Semilla S1: regla local mínima con mezcla 3x3 y compuertas suaves."""
    def __init__(self, cfg:Config, pesos=None, sesgo=None, rng=None):
        self.cfg = cfg
        # RNG reproducible (NumPy)
        self.rng = np.random.default_rng(cfg.semilla_global) if rng is None else rng
        self._step = 0
        C = cfg.canales
        if pesos is None:
            self.pesos = self.rng.normal(0, cfg.escala_init, size=(3,3,C,C)).astype(np.float32)
            self.sesgo  = self.rng.normal(0, cfg.escala_init, size=(C,)).astype(np.float32)
        else:
            self.pesos = pesos.astype(np.float32)
            self.sesgo = sesgo.astype(np.float32)
        self.mask_async = None

    def init_estado(self):
        H,W,C = self.cfg.alto, self.cfg.ancho, self.cfg.canales
        est = np.zeros((H,W,C), dtype=np.float32)
        # semilla mínima: 1 píxel con masa y señal
        for _ in range(self.cfg.semillas):
            y = self.rng.integers(0,H); x = self.rng.integers(0,W)
            est[y,x,0] = 1.0   # masa
            est[y,x,1] = 0.5   # señal
        return est

    def _conv3x3(self, est):
        H,W,C = est.shape
        out = np.zeros_like(est, dtype=np.float32)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                v = np.roll(est, shift=(dy,dx), axis=(0,1))
                out += np.tensordot(v, self.pesos[dy+1,dx+1], axes=([2],[0]))
        out += self.sesgo
        return out

    def paso(self, est, estimulo=None):
        H,W,C = est.shape
        # 1) estimulación externa → canal modulador (3)
        if estimulo is not None:
            y,x,intensidad,radio = estimulo
            yy,xx = np.ogrid[:H,:W]
            dy = np.minimum(np.abs(yy-y), H-np.abs(yy-y))
            dx = np.minimum(np.abs(xx-x), W-np.abs(xx-x))
            mask = ((dy*dy + dx*dx) <= radio*radio).astype(np.float32)
            est[...,3] += intensidad*mask

        pre = self._conv3x3(est)     # mezcla local
        m = sigmoid(est[...,3:4])    # puerta moduladora local
        delta = tanh(pre) * m        # cambio propuesto

        # ASINCRÓNICO: actualiza solo una fracción de celdas
        if self.mask_async is None:
            prob = self.cfg.frac_async
            self.mask_async = (self.rng.random((H,W,1)) < prob).astype(np.float32)
        elif self.cfg.async_refresh_every>0 and (self._step % self.cfg.async_refresh_every == 0):
            prob = self.cfg.frac_async
            self.mask_async = (self.rng.random((H,W,1)) < prob).astype(np.float32)
        nuevo = est + self.cfg.paso * delta * self.mask_async

        # límites suaves
        nuevo[...,0] = np.clip(nuevo[...,0], 0.0, 1.0)   # masa
        for c in (1,2,3):
            nuevo[...,c] = np.clip(nuevo[...,c], -3.0, 3.0)
        # fugas leves
        nuevo[...,1] = 0.995*nuevo[...,1]   # señal se atenúa
        nuevo[...,2] = 0.999*nuevo[...,2]   # memoria casi persistente
        nuevo[...,3] = 0.98*nuevo[...,3]    # modulador decae
        self._step += 1
        return nuevo


# =============================
# Backend alternativo: PyTorch
# =============================
class S1Torch:
    """Semilla S1 con PyTorch (opcional GPU). Misma interfaz que S1."""
    def __init__(self, cfg:Config, pesos=None, sesgo=None, rng=None, device=None):
        assert TORCH_AVAILABLE, "PyTorch no disponible"
        self.cfg = cfg
        self.device = device or ("cuda" if TORCH_HAS_CUDA else "cpu")
        # Reproducibilidad
        g = torch.Generator(device=self.device)
        if isinstance(rng, torch.Generator):
            g = rng
        elif isinstance(rng, (int, np.integer)):
            g.manual_seed(int(rng))
        else:
            g.manual_seed(cfg.semilla_global)
        self.g = g
        C = cfg.canales
        if pesos is None:
            self.pesos = torch.normal(
                mean=0.0, std=cfg.escala_init,
                size=(C, C, 3, 3), generator=g, device=self.device, dtype=torch.float32
            )
            self.sesgo = torch.normal(
                mean=0.0, std=cfg.escala_init,
                size=(C,), generator=g, device=self.device, dtype=torch.float32
            )
        else:
            # Acepta numpy arrays
            self.pesos = torch.as_tensor(pesos, device=self.device).permute(3,2,0,1).contiguous()
            self.sesgo = torch.as_tensor(sesgo, device=self.device)
        self.mask_async = None
        self._step = 0

    def init_estado(self):
        H,W,C = self.cfg.alto, self.cfg.ancho, self.cfg.canales
        est = torch.zeros((H,W,C), dtype=torch.float32, device=self.device)
        # semilla mínima
        for _ in range(self.cfg.semillas):
            y = torch.randint(0, H, (1,), generator=self.g, device=self.device).item()
            x = torch.randint(0, W, (1,), generator=self.g, device=self.device).item()
            est[y,x,0] = 1.0
            est[y,x,1] = 0.5
        return est

    def _conv3x3(self, est):
        # est: HxWxC → 1xCxHxW con padding circular
        x = est.permute(2,0,1).unsqueeze(0)
        x = F.pad(x, (1,1,1,1), mode='circular')
        # pesos: CxCx3x3 (ya en formato conv)
        y = F.conv2d(x, self.pesos, bias=self.sesgo, stride=1, padding=0)
        y = y.squeeze(0).permute(1,2,0)
        return y

    def paso(self, est, estimulo=None):
        H,W,C = est.shape
        if estimulo is not None:
            y,x,intensidad,radio = estimulo
            yy = torch.arange(H, device=self.device)[:,None]
            xx = torch.arange(W, device=self.device)[None,:]
            dy = torch.minimum((yy - y).abs(), torch.tensor(H, device=self.device) - (yy - y).abs())
            dx = torch.minimum((xx - x).abs(), torch.tensor(W, device=self.device) - (xx - x).abs())
            mask = ((dy*dy + dx*dx) <= radio*radio).float()
            est[...,3] = est[...,3] + intensidad*mask

        pre = self._conv3x3(est)
        m = torch.sigmoid(est[...,3:4])
        delta = torch.tanh(pre) * m

        # ASINCRONÍA
        if self.mask_async is None:
            prob = self.cfg.frac_async
            self.mask_async = (torch.rand((H,W,1), generator=self.g, device=self.device) < prob).float()
        elif self.cfg.async_refresh_every>0 and (self._step % self.cfg.async_refresh_every == 0):
            prob = self.cfg.frac_async
            self.mask_async = (torch.rand((H,W,1), generator=self.g, device=self.device) < prob).float()
        nuevo = est + self.cfg.paso * delta * self.mask_async

        # límites suaves y fugas
        nuevo_masa = torch.clamp(nuevo[...,0], 0.0, 1.0)
        nuevo_s1 = torch.clamp(nuevo[...,1], -3.0, 3.0) * 0.995
        nuevo_s2 = torch.clamp(nuevo[...,2], -3.0, 3.0) * 0.999
        nuevo_mod= torch.clamp(nuevo[...,3], -3.0, 3.0) * 0.98
        nuevo = torch.stack([nuevo_masa, nuevo_s1, nuevo_s2, nuevo_mod], dim=-1)
        self._step += 1
        return nuevo

# ------------------------------
# Evaluación: VIABILIDADES + descriptores
# ------------------------------
def evalua_genoma(pesos, sesgo, cfg:Config, rng):
    # Selección de backend con fallback seguro
    requested_torch = (cfg.backend == "torch")
    use_torch = ((cfg.backend in ("auto","torch")) and TORCH_AVAILABLE)
    if use_torch:
        # Si hay rng numpy, úsalo para derivar una seed
        seed = cfg.semilla_global
        try:
            if hasattr(rng, 'integers'):
                seed = int(rng.integers(0, 2**31-1))
        except Exception:
            pass
        s1 = S1Torch(cfg, pesos=pesos, sesgo=sesgo, rng=seed)
    else:
        s1 = S1(cfg, pesos=pesos, sesgo=sesgo, rng=rng)
    est = s1.init_estado()
    H,W,C = est.shape
    historia = []
    poke = (H//2, W//2, cfg.poke_intensidad, cfg.poke_radio)

    t0 = int(cfg.warmup) if cfg.warmup>0 else cfg.pasos//2
    for t in range(cfg.pasos):
        est = s1.paso(est, estimulo=poke if t==t0 else None)
        if TORCH_AVAILABLE and isinstance(s1, S1Torch):
            historia.append(est[...,0].detach().cpu().numpy().copy())
        else:
            historia.append(est[...,0].copy())
    hist = np.stack(historia, axis=0)  # T x H x W (numpy para métricas/plots)

    # ----- Métricas densas -----
    dens_t = hist.mean(axis=(1,2))
    densidad_final = float(dens_t[-1])
    # oscilación (varianza temporal de densidad final 1/3 de los últimos pasos)
    tail = dens_t[-(cfg.pasos//3):]
    oscil = float(np.var(tail))
    # respuesta al poke: promedio de |cambio pixel a pixel| sobre una ventana post-poke
    t0 = int(cfg.warmup) if cfg.warmup>0 else cfg.pasos//2
    w = max(1, int(cfg.resp_ventana))
    base = hist[t0-1]
    post = hist[t0+1:t0+1+w]
    respuesta = float(np.mean(np.abs(post - base)))  # [0,1]
    # no-extinción/no-explosión (fracciones)
    frac_vivo = float((hist[-1] > 0.1).mean())

    # ----- Viabilidades (lexicográficas) -----
    ok_no_extincion = frac_vivo >= cfg.th_no_extincion_min
    ok_no_explosion = frac_vivo <= cfg.th_no_explosion_max
    ok_estabilidad  = oscil <= cfg.th_oscil_max
    ok_respuesta    = respuesta >= cfg.th_resp_min
    viable = ok_no_extincion and ok_no_explosion and ok_estabilidad and ok_respuesta

    # Descriptores para QD: (densidad, respuesta, computo)
    comp = 0.0
    if viable:
        # cómputo ligero (reservorio) – barato
        comp = _score_reservorio_ligero(pesos, sesgo, cfg, rng)
    desc = (densidad_final, respuesta, oscil, comp)
    return viable, desc, hist

def evalua_genoma_avg(pesos, sesgo, cfg:Config, seeds:int):
    """Evalúa varias semillas y promedia; devuelve hist de la mejor respuesta para visualización."""
    vals = []
    best = None
    for k in range(seeds):
        rng_k = np.random.default_rng(cfg.semilla_global + 1000 + k)
        ok, desc, hist = evalua_genoma(pesos, sesgo, cfg, rng_k)
        vals.append((ok, desc, hist))
        if (best is None) or (desc[1] > best[1][1]):
            best = (ok, desc, hist)
    oks = [1 if v[0] else 0 for v in vals]
    viable = (sum(oks) >= (len(oks)//2 + 1))
    dens = float(np.mean([v[1][0] for v in vals]))
    resp = float(np.mean([v[1][1] for v in vals]))
    osc  = float(np.mean([v[1][2] for v in vals]))
    comp = float(np.mean([v[1][3] for v in vals]))
    return viable, (dens, resp, osc, comp), best[2]

# ------------------------------
# Evaluación en lote (torch)
# ------------------------------
def evalua_genomas_batch_torch(pesos_list, sesgo_list, cfg:Config):
    """Evalúa un lote de genomas con torch. Devuelve listas de (viable, desc, hist).
       Asume todos con igual forma (3x3xC x C) y mismo cfg.
    """
    assert TORCH_AVAILABLE, "PyTorch no disponible"
    device = "cuda" if TORCH_HAS_CUDA else "cpu"
    C = cfg.canales; H = cfg.alto; W = cfg.ancho
    B = len(pesos_list)
    # Tensores
    pesos = torch.stack([torch.as_tensor(p, device=device).permute(3,2,0,1) for p in pesos_list], dim=0)  # BxCxCx3x3
    sesgo  = torch.stack([torch.as_tensor(s, device=device) for s in sesgo_list], dim=0)                    # BxC
    # Estados iniciales
    est = torch.zeros((B,H,W,C), device=device)
    # Semillas aleatorias diferenciadas por batch
    for b in range(B):
        y = torch.randint(0, H, (cfg.semillas,), device=device)
        x = torch.randint(0, W, (cfg.semillas,), device=device)
        est[b,y,x,0] = 1.0
        est[b,y,x,1] = 0.5
    # Máscara async por batch
    mask_async = torch.rand((B,H,W,1), device=device) < cfg.frac_async
    # Historial de masa
    hist_list = []
    t0 = int(cfg.warmup) if cfg.warmup>0 else cfg.pasos//2
    poke_y = H//2; poke_x = W//2
    for t in range(cfg.pasos):
        # estímulo
        if t == t0:
            yy = torch.arange(H, device=device)[None,:,None]
            xx = torch.arange(W, device=device)[None,None,:]
            dy = torch.minimum((yy - poke_y).abs(), torch.tensor(H, device=device) - (yy - poke_y).abs())
            dx = torch.minimum((xx - poke_x).abs(), torch.tensor(W, device=device) - (xx - poke_x).abs())
            disk = ((dy*dy + dx*dx) <= cfg.poke_radio*cfg.poke_radio).float()
            est[...,3] = est[...,3] + cfg.poke_intensidad * disk
        # conv por batch: desdoblamos en B grupos (grouped conv)
        x = est.permute(0,3,1,2)  # BxCxHxW
        x = F.pad(x, (1,1,1,1), mode='circular')
        # Empaquetar batch en el canal para grouped conv
        x = x.reshape(1, B*C, H+2, W+2)
        w = pesos.view(B*C, C, 3, 3)
        b = sesgo.view(B*C)
        y = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=B)
        y = y.view(B, C, H, W).permute(0,2,3,1)
        m = torch.sigmoid(est[...,3:4])
        delta = torch.tanh(y) * m
        # refresco determinista de máscara async
        if cfg.async_refresh_every>0 and (t % cfg.async_refresh_every == 0):
            mask_async = (torch.rand((B,H,W,1), device=device) < cfg.frac_async)
        est = est + cfg.paso * delta * mask_async
        # límites/fugas
        est[...,0] = torch.clamp(est[...,0], 0.0, 1.0)
        for c in (1,2,3):
            est[...,c] = torch.clamp(est[...,c], -3.0, 3.0)
        est[...,1] = 0.995*est[...,1]
        est[...,2] = 0.999*est[...,2]
        est[...,3] = 0.98*est[...,3]
        hist_list.append(est[...,0].detach().cpu())
    hist = torch.stack(hist_list, dim=0).numpy()  # T x B x H x W
    # Métricas por batch (numpy para reusar código)
    results = []
    for b in range(B):
        hb = hist[:,b]
        dens_t = hb.mean(axis=(1,2))
        densidad_final = float(dens_t[-1])
        oscil = float(np.var(dens_t[-(cfg.pasos//3):]))
        base = hb[t0-1]
        post = hb[t0+1:t0+1+max(1,cfg.resp_ventana)]
        respuesta = float(np.mean(np.abs(post - base)))
        frac_vivo = float((hb[-1] > 0.1).mean())
        ok_no_ext = frac_vivo >= cfg.th_no_extincion_min
        ok_no_exp = frac_vivo <= cfg.th_no_explosion_max
        ok_estab  = oscil <= cfg.th_oscil_max
        ok_resp   = respuesta >= cfg.th_resp_min
        viable = ok_no_ext and ok_no_exp and ok_estab and ok_resp
        # Por coste, omitimos cómputo en ruta batch (0.0)
        desc = (densidad_final, respuesta, oscil, 0.0)
        results.append((viable, desc, hb))
    return results

# ------------------------------
# QD: llenar mapa por descriptores (ρ,Δ)
# ------------------------------
def binar_desc(dens, resp, comp, bins=10, resp_ref=0.06, bins_comp=6):
    i = int(np.clip(dens*bins, 0, bins-1))
    j = int(np.clip((resp/resp_ref)*bins, 0, bins-1))
    k = int(np.clip(comp*bins_comp, 0, bins_comp-1))
    return i,j,k

def nuevo_genoma(rng, C, escala):
    pesos = rng.normal(0, escala, size=(3,3,C,C)).astype(np.float32)
    sesgo  = rng.normal(0, escala, size=(C,)).astype(np.float32)
    return pesos, sesgo

def qd_busqueda(cfg:Config, log=None):
    rng = np.random.default_rng(cfg.semilla_global)
    C = cfg.canales
    mapa = np.full((cfg.bins, cfg.bins, cfg.bins_comp), np.nan, dtype=np.float32)   # NaN = vacío
    elites = {}  # (i,j,k)->(pesos,sesgo)
    elite_desc = {}  # (i,j,k)-> desc (dens, resp, osc, comp)
    elite_speed = {}  # (i,j)-> velocidad_frente

    # --- helpers: frontera y mutación ---
    def vecinos3(i,j,k, B1, B2, B3):
        for di in (-1,0,1):
            for dj in (-1,0,1):
                for dk in (-1,0,1):
                    if di==0 and dj==0 and dk==0: continue
                    ii = i+di; jj = j+dj; kk = k+dk
                    if 0<=ii<B1 and 0<=jj<B2 and 0<=kk<B3:
                        yield ii,jj,kk

    def frontera_elites():
        B1,B2,B3 = cfg.bins, cfg.bins, cfg.bins_comp
        f = []
        for ijk in elites.keys():
            i,j,k = ijk
            for ii,jj,kk in vecinos3(i,j,k,B1,B2,B3):
                if np.isnan(mapa[ii,jj,kk]):
                    f.append(ijk)
                    break
        return f

    def muta_genoma(p, s, rng,
                     sigma_small=0.03, sigma_big=None,
                     p_big=0.20, p_sparse=0.60, frac_sparse=0.10):
        sigma_b = sigma_big if sigma_big is not None else max(0.08, sigma_small*3.0)
        use_big = (rng.random() < p_big)
        sig = sigma_b if use_big else sigma_small
        # esparso o denso
        if rng.random() < p_sparse:
            mask_w = (rng.random(p.shape) < frac_sparse).astype(np.float32)
            mask_b = (rng.random(s.shape) < frac_sparse).astype(np.float32)
            dp = rng.normal(0, sig, size=p.shape).astype(np.float32) * mask_w
            ds = rng.normal(0, sig, size=s.shape).astype(np.float32) * mask_b
        else:
            dp = rng.normal(0, sig, size=p.shape).astype(np.float32)
            ds = rng.normal(0, sig, size=s.shape).astype(np.float32)
        return (p + dp).astype(np.float32), (s + ds).astype(np.float32)

    def _log(msg):
        if log:
            log(msg)
        else:
            print(msg)

    start = datetime.now()
    total_evals = 0
    block_evals = 0
    block_accepts = 0
    block_newcells = 0
    coverage_hist = []

    # arranque
    for _ in range(12):
        p,s = nuevo_genoma(rng, C, cfg.escala_init)
        ok, desc, hist = evalua_genoma(p,s,cfg,rng)
        if ok:
            i,j,k = binar_desc(desc[0], desc[1], desc[3], cfg.bins, cfg.resp_ref, cfg.bins_comp)
            # score de complejidad
            ent = entropia_01(hist[-1])
            edges = densidad_bordes(hist[-1])
            osc = desc[2]
            score = ent + edges - 5.0*max(0.0, osc - 0.03)
            if np.isnan(mapa[i,j,k]) or score > mapa[i,j,k]:
                mapa[i,j,k] = score
                elites[(i,j,k)] = (p,s)
                elite_desc[(i,j,k)] = desc
                # velocidad del frente (solo informativa)
                v = velocidad_frente(hist, cfg.pasos//2, cfg.resp_ventana)
                elite_speed[(i,j,k)] = v
                block_accepts += 1
            total_evals += 1; block_evals += 1

    # Mutación auto-adaptativa (1/5 rule)
    sigma = 0.03
    explore_p = 0.20
    immigrants_p = 0.05
    success_window = []  # 1 éxito, 0 fallo

    for it in range(cfg.iteraciones):
        if elites:
            # Selección pro-frontera: si hay élites con vecinos vacíos, priorízalas
            fr = frontera_elites()
            if fr:
                ij = fr[rng.integers(0, len(fr))]
            else:
                ij = list(elites.keys())[rng.integers(0,len(elites))]
            p,s = elites[ij]
            # Inmigrantes aleatorios pequeños + exploración adaptativa (usar el máx)
            p_new = max(immigrants_p, explore_p)
            explore = (rng.random() < p_new)
            # ¿batch?
            if (cfg.backend in ("auto","torch")) and TORCH_AVAILABLE and cfg.batch > 1:
                B = cfg.batch
                pesos_batch = []
                sesgo_batch = []
                for _ in range(B):
                    if explore:
                        pb, sb = nuevo_genoma(rng, C, cfg.escala_init)
                    else:
                        pb, sb = muta_genoma(p, s, rng, sigma_small=sigma)
                    pesos_batch.append(pb)
                    sesgo_batch.append(sb)
                batch_res = evalua_genomas_batch_torch(pesos_batch, sesgo_batch, cfg)
                candidatos = []
                for idx,(ok, desc, hist) in enumerate(batch_res):
                    if ok:
                        candidatos.append((idx, desc, hist))
                if not candidatos:
                    success_window.append(0)
                    continue
                # elige el mejor por score QD
                def score_from(desc, hist):
                    ent = entropia_01(hist[-1])
                    edges = densidad_bordes(hist[-1])
                    osc = desc[2]
                    return ent + edges - 5.0*max(0.0, osc - 0.03)
                best_idx, best_desc, best_hist = max(candidatos, key=lambda t: score_from(t[1], t[2]))
                # Sustituir p,s por el del mejor del batch
                p = pesos_batch[best_idx]
                s = sesgo_batch[best_idx]
                success_window.append(1)
            else:
                if explore:
                    p, s = nuevo_genoma(rng, C, cfg.escala_init)
                else:
                    p, s = muta_genoma(p, s, rng, sigma_small=sigma)
        else:
            p,s = nuevo_genoma(rng, C, cfg.escala_init)

        # Evaluación: multi-semilla si se solicita (solo ruta no-batch)
        if cfg.avg_seeds > 1:
            ok, desc, hist = evalua_genoma_avg(p, s, cfg, cfg.avg_seeds)
        else:
            ok, desc, hist = evalua_genoma(p,s,cfg,rng)
        if ok:
            i,j,k = binar_desc(desc[0], desc[1], desc[3], cfg.bins, cfg.resp_ref, cfg.bins_comp)
            ent = entropia_01(hist[-1])
            edges = densidad_bordes(hist[-1])
            osc = desc[2]
            score = ent + edges - 5.0*max(0.0, osc - 0.03)
            # QD puro: llenar primero; reemplazo aleatorio raro (p=0.05) para evitar congelación
            t0 = int(cfg.warmup) if cfg.warmup>0 else cfg.pasos//2
            if np.isnan(mapa[i,j,k]):
                mapa[i,j,k] = score
                elites[(i,j,k)] = (p,s)
                elite_desc[(i,j,k)] = desc
                v = velocidad_frente(hist, t0, cfg.resp_ventana)
                elite_speed[(i,j,k)] = v
                block_accepts += 1
                block_newcells += 1
            else:
                # Reemplazo lexicográfico con prioridad a cómputo
                do_replace = False
                if getattr(cfg, 'comp_priority', False) and (i,j,k) in elite_desc:
                    old_comp = float(elite_desc[(i,j,k)][3])
                    if desc[3] > old_comp + float(getattr(cfg, 'comp_priority_eps', 0.03)):
                        do_replace = True
                # Si no mejora comp de forma clara, aceptar si mejora score o por rare replace
                if (not do_replace) and (not np.isnan(mapa[i,j,k])) and (score > float(mapa[i,j,k]) + 1e-9):
                    do_replace = True
                if (not do_replace) and (rng.random() < 0.05):
                    do_replace = True
                if do_replace:
                    mapa[i,j,k] = score
                    elites[(i,j,k)] = (p,s)
                    elite_desc[(i,j,k)] = desc
                    v = velocidad_frente(hist, t0, cfg.resp_ventana)
                    elite_speed[(i,j,k)] = v
                    block_accepts += 1
            success_window.append(1)
        else:
            success_window.append(0)
        total_evals += 1; block_evals += 1

        # Adaptación de sigma cada 40 intentos
        if len(success_window) >= 40:
            rate = sum(success_window[-40:]) / 40.0
            if rate > 0.25:
                sigma *= 1.2
            elif rate < 0.15:
                sigma *= 0.85
            sigma = float(np.clip(sigma, 0.01, 0.15))
            # Exploración adaptativa
            if rate < 0.10:
                explore_p = 0.35
            elif rate > 0.25:
                explore_p = 0.20

        # Reporte periódico
        if (it+1) % max(1,cfg.log_every) == 0:
            cobertura = len(elites)
            if cobertura>0:
                mapa_safe = np.where(np.isnan(mapa), -np.inf, mapa)
                best_score = float(np.max(mapa_safe))
            else:
                best_score = float('nan')
            if cobertura>0:
                mapa_safe = np.where(np.isnan(mapa), -np.inf, mapa)
                best_ij = np.unravel_index(int(np.argmax(mapa_safe)), mapa.shape)
                best_desc = elite_desc.get(best_ij, (float('nan'),)*4)
            else:
                best_ij = None; best_desc=(float('nan'),)*4
            mean_resp = float(np.mean([d[1] for d in elite_desc.values()])) if elite_desc else float('nan')
            mean_speed = float(np.mean(list(elite_speed.values()))) if elite_speed else float('nan')
            median_resp = float(np.median([d[1] for d in elite_desc.values()])) if elite_desc else float('nan')
            median_osc = float(np.median([d[2] for d in elite_desc.values()])) if elite_desc else float('nan')
            mean_comp = float(np.mean([d[3] for d in elite_desc.values()])) if elite_desc else float('nan')
            median_comp = float(np.median([d[3] for d in elite_desc.values()])) if elite_desc else float('nan')
            good_thr = float(getattr(cfg, 'comp_acc_thr', 0.7))
            good_comp_ratio = float(np.mean([1.0 if d[3] >= good_thr else 0.0 for d in elite_desc.values()])) if elite_desc else float('nan')
            elapsed = (datetime.now()-start).total_seconds()
            acc_rate = (block_accepts/max(1,block_evals))
            novelty_rate = (block_newcells/max(1,block_evals))
            fr_cells = frontera_elites()
            frontier_ratio = (len(fr_cells)/max(1,cobertura)) if cobertura>0 else float('nan')
            coverage_hist.append(cobertura)
            _log(
                f"it {it+1}/{cfg.iteraciones} | elites {cobertura}/{cfg.bins*cfg.bins*cfg.bins_comp} "
                f"({cobertura/(cfg.bins*cfg.bins):.2%}) | acc_blk {block_accepts}/{block_evals} ({acc_rate:.1%}) | new_blk {block_newcells}/{block_evals} ({novelty_rate:.1%}) | fr_ratio {frontier_ratio:.2f} | "
                f"best {best_score:.3f} @ {best_ij} desc(d,Δ,σ,comp)={tuple(round(x,4) for x in best_desc)} | "
                f"Δ̄={mean_resp:.4f} medΔ={median_resp:.4f} medσ={median_osc:.4f} comp̄={mean_comp:.3f} medComp={median_comp:.3f} goodComp%={good_comp_ratio:.1%} (thr={good_thr:.2f}) | v̄={mean_speed:.4f} | t={elapsed:.1f}s"
            )
            block_evals = 0; block_accepts = 0; block_newcells = 0

    return mapa, elites, coverage_hist

def demo():
    # CLI rápida
    parser = argparse.ArgumentParser(description="Semilla S1 - CA continuo + QD")
    # Presets de conveniencia: establecen buenos valores por defecto según el objetivo.
    parser.add_argument("--preset", choices=["rapido","estandar","exhaustivo"], default="estandar",
                        help="Perfil de ejecución: rapido (sanity check), estandar (buena calidad por defecto), exhaustivo (mejor cómputo y cobertura)")
    parser.add_argument("--backend", choices=["auto","numpy","torch"], default="torch")
    parser.add_argument("--iter", type=int, default=None, help="Iteraciones de QD")
    parser.add_argument("--bins", type=int, default=None, help="Resolución del mapa QD (densidad y respuesta)")
    parser.add_argument("--bins-comp", type=int, default=None, help="Resolución del eje de cómputo")
    parser.add_argument("--pasos", type=int, default=None, help="Pasos de simulación por evaluación")
    parser.add_argument("--poke-int", type=float, default=None, help="Intensidad del estímulo")
    parser.add_argument("--resp-win", type=int, default=None, help="Ventana de respuesta tras el poke")
    parser.add_argument("--th-osc", type=float, default=None, help="Umbral máximo de oscilación")
    parser.add_argument("--th-resp", type=float, default=None, help="Umbral mínimo de respuesta")
    parser.add_argument("--resp-ref", type=float, default=None, help="Escala de referencia para el eje Δ del mapa QD")
    parser.add_argument("--batch", type=int, default=None, help="Tamaño de lote para evaluación paralela (torch)")
    parser.add_argument("--async-refresh-every", type=int, default=None, help="Refrescar máscara asíncrona cada K pasos (0=constante)")
    parser.add_argument("--warmup", type=int, default=None, help="Paso del poke (0 = pasos//2)")
    parser.add_argument("--avg-seeds", type=int, default=None, help="Número de semillas por evaluación para promediar (ruta no-batch)")
    parser.add_argument("--anim", type=str, default="auto", help="Ruta de salida de animación GIF/MP4 del mejor élite (auto = S1_anim_<timestamp>.mp4; vacío = no guardar)")
    parser.add_argument("--anim-fps", type=int, default=12, help="FPS para la animación del mejor élite")
    parser.add_argument("--save-hist", type=str, default="", help="Guardar hist del mejor élite como .npy (vacío = no guardar - se calcula on-demand)")
    parser.add_argument("--topk-comp", type=int, default=0, help="Listar/guardar los K mejores por 'comp' (0=desactivado)")
    parser.add_argument("--save-topk", type=str, default="", help="Ruta .npz para guardar los top-K por cómputo (vacío = no guardar - semilla_elites.npz ya es suficiente)")
    parser.add_argument("--comp-T", type=int, default=None, help="Pasos del test de cómputo ligero")
    parser.add_argument("--comp-embed", type=int, default=None, help="Embedding temporal para cómputo")
    parser.add_argument("--save-best", type=str, default="semilla_elites.npz", help="Ruta .npz para guardar SIEMPRE el mejor por cómputo (vacío = no guardar)")
    args = parser.parse_args()

    # Aplicar presets (valores por defecto del perfil) respetando overrides explícitos
    preset = args.preset
    # Defaults base por perfil
    if preset == "rapido":
        defaults = dict(iteraciones=120, bins=8, bins_comp=5, pasos=80, poke_intensidad=2.0,
                        resp_ventana=5, th_oscil_max=0.06, th_resp_min=0.02, resp_ref=0.06,
                        batch=1, async_refresh_every=0, warmup=0, avg_seeds=1,
                        comp_T=120, comp_embed=8, topk_comp=0)
    elif preset == "full":
        defaults = dict(iteraciones=1200, bins=12, bins_comp=8, pasos=200, poke_intensidad=2.5,
                        resp_ventana=8, th_oscil_max=0.05, th_resp_min=0.03, resp_ref=0.06,
                        batch=1, async_refresh_every=1, warmup=30, avg_seeds=1,
                        comp_T=200, comp_embed=12, topk_comp=3)
    else:  # estandar
        defaults = dict(iteraciones=400, bins=10, bins_comp=6, pasos=150, poke_intensidad=2.5,
                        resp_ventana=6, th_oscil_max=0.05, th_resp_min=0.03, resp_ref=0.06,
                        batch=1, async_refresh_every=1, warmup=20, avg_seeds=1,
                        comp_T=160, comp_embed=10, topk_comp=2)

    # Helper: usar arg si no es None; si es None, usar default del preset; si tampoco, caer a Config default
    def pick(val, key, fallback):
        return fallback if val is None else val

    cfg = Config(
        backend=args.backend,
        iteraciones=pick(args.iter, 'iteraciones', defaults.get('iteraciones', 220)),
        bins=pick(args.bins, 'bins', defaults.get('bins', 10)),
        bins_comp=pick(args.bins_comp, 'bins_comp', defaults.get('bins_comp', 6)),
        pasos=pick(args.pasos, 'pasos', defaults.get('pasos', 70)),
        poke_intensidad=pick(args.poke_int, 'poke_intensidad', defaults.get('poke_intensidad', 2.5)),
        resp_ventana=pick(args.resp_win, 'resp_ventana', defaults.get('resp_ventana', 5)),
        th_oscil_max=pick(args.th_osc, 'th_oscil_max', defaults.get('th_oscil_max', 0.05)),
        th_resp_min=pick(args.th_resp, 'th_resp_min', defaults.get('th_resp_min', 0.03)),
        resp_ref=pick(args.resp_ref, 'resp_ref', defaults.get('resp_ref', 0.06)),
        batch=pick(args.batch, 'batch', defaults.get('batch', 1)),
        log_every=10,
        async_refresh_every=pick(args.async_refresh_every, 'async_refresh_every', defaults.get('async_refresh_every', 0)),
        warmup=pick(args.warmup, 'warmup', defaults.get('warmup', 0)),
        avg_seeds=pick(args.avg_seeds, 'avg_seeds', defaults.get('avg_seeds', 1))
    )
    # Ajustes de cómputo desde CLI
    cfg.comp_T = pick(args.comp_T, 'comp_T', defaults.get('comp_T', cfg.comp_T))
    cfg.comp_embed = pick(args.comp_embed, 'comp_embed', defaults.get('comp_embed', cfg.comp_embed))
    # Ajuste de topk_comp efectivo por preset (usado más abajo)
    if args.topk_comp is None:
        try:
            args.topk_comp = int(defaults.get('topk_comp', 0))
        except Exception:
            args.topk_comp = 0
    # Logger a archivo + consola
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join('logs', f'S1_run_{ts}.log')
    def logger(msg):
        print(msg)
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(str(msg)+"\n")
        except Exception:
            pass
    # Valor por defecto: si --anim=="auto", guardamos un MP4 con timestamp
    if args.anim == "auto":
        args.anim = f"semilla_animacion.mp4"
    # Valor por defecto: si --save-hist=="auto", guardamos un .npy sin timestamp (estable)
    if args.save_hist == "auto":
        args.save_hist = ""  # Por defecto no guardar historial separado (ya está en memoria para análisis)
    # Reporte de backend efectivo (una sola vez)
    eff_torch = ((cfg.backend in ("auto","torch")) and TORCH_AVAILABLE)
    if eff_torch:
        device = "cuda" if (TORCH_AVAILABLE and TORCH_HAS_CUDA) else "cpu"
        print(f"Backend: torch ({device})")
    else:
        if cfg.backend == "torch" and not TORCH_AVAILABLE:
            print("Backend: torch solicitado pero no disponible → usando numpy")
        else:
            print("Backend: numpy")
    logger(f"CFG(preset={preset}): bins={cfg.bins}, bins_comp={cfg.bins_comp}, iter={cfg.iteraciones}, pasos={cfg.pasos}, batch={cfg.batch}, resp_ref={cfg.resp_ref}, th_resp_min={cfg.th_resp_min}, th_osc={cfg.th_oscil_max}, warmup={cfg.warmup}, avg_seeds={cfg.avg_seeds}, async_refresh_every={cfg.async_refresh_every}, comp_T={cfg.comp_T}, comp_embed={cfg.comp_embed}")
    mapa, elites, coverage_hist = qd_busqueda(cfg, log=logger)
    cobertura = len(elites)
    print("Celdas con élite:", cobertura, "de", cfg.bins*cfg.bins*cfg.bins_comp)

    # Visualización/animación por defecto: mejor por 'comp' (capacidad de cómputo)
    if elites:
        # Re-evaluar descriptores para todos los élites una sola vez (para comp consistente)
        pares = []  # (ijk, (p,s), desc, hist_opt)
        base_seed = cfg.semilla_global + 100
        for n, (ijk, (p,s)) in enumerate(elites.items()):
            ok_k, d_k, _ = evalua_genoma(p, s, cfg, np.random.default_rng(base_seed + n))
            pares.append((ijk, (p,s), d_k))

        # Mejor por cómputo
        ijk_comp, (p_comp, s_comp), d_comp = max(pares, key=lambda t: t[2][3])
        ok, desc_comp, hist = evalua_genoma(p_comp, s_comp, cfg, np.random.default_rng(cfg.semilla_global+1))
        v_best = velocidad_frente(hist, cfg.pasos//2, cfg.resp_ventana)
        print("Mejor celda (cómputo):", ijk_comp, "desc=(dens,resp,osc,comp)≈", tuple(round(x,4) for x in desc_comp), f"v≈{v_best:.4f}")
        # Guardar SIEMPRE el mejor por cómputo con nombre estandarizado (si no se desactiva)
        if args.save_best:
            try:
                np.savez(args.save_best, pesos=p_comp, sesgo=s_comp, cfg=cfg.__dict__)
                print(f"Mejor por cómputo guardado en {args.save_best}")
            except Exception as e:
                print(f"No se pudo guardar save-best ({args.save_best}): {e}")

        # También reportar la mejor por score-QD (mapa)
        best_ijk_qd = max(elites.keys(), key=lambda ijk: mapa[ijk])
        p_qd, s_qd = elites[best_ijk_qd]
        ok_qd, desc_qd, _ = evalua_genoma(p_qd, s_qd, cfg, np.random.default_rng(cfg.semilla_global+2))
        print("Mejor celda (score-QD):", best_ijk_qd, "desc=(dens,resp,osc,comp)≈", tuple(round(x,4) for x in desc_qd))

        # alternativa ‘equilibrada’: baja oscilación + densidad ~0.5 + buena respuesta
        def score_equilibrio(item):
            ijk,(p,s) = item
            ok, d, _ = evalua_genoma(p,s,cfg, np.random.default_rng(cfg.semilla_global+3))
            dens, resp, osc, comp = d
            return (osc, abs(dens-0.5), -resp)
        best_eq_ijk, (p_eq, s_eq) = min(elites.items(), key=score_equilibrio)
        ok, desc_eq, hist_eq = evalua_genoma(p_eq, s_eq, cfg, np.random.default_rng(cfg.semilla_global+4))
        v_eq = velocidad_frente(hist_eq, cfg.pasos//2, cfg.resp_ventana)
        print("Mejor celda (equilibrada):", best_eq_ijk, "desc=(dens,resp,osc,comp)≈", tuple(round(x,4) for x in desc_eq), f"v≈{v_eq:.4f}")

        # Figura MEJORADA: mapa QD proyectado (máx sobre eje cómputo)
        plt.figure(figsize=(10, 4))

        # Panel izquierdo: Mapa QD con score-QD
        ax1 = plt.subplot(1, 2, 1)
        mapa_safe = np.where(np.isnan(mapa), -np.inf, mapa)
        M = np.max(mapa_safe, axis=2)
        M = np.ma.masked_less(M, -1e10)
        
        # Usar colormap más claro y contrastado
        im1 = ax1.imshow(M.T, origin='lower', cmap='plasma', aspect='auto')
        
        # Ejes más informativos
        ax1.set_title("Mapa QD: Score-QD\n(máx sobre eje cómputo)", fontsize=11, pad=10)
        
        # Convertir bins a valores reales aproximados
        x_ticks = np.linspace(0, cfg.bins-1, 5)
        y_ticks = np.linspace(0, cfg.bins-1, 5)
        x_labels = [f"{i*1.0/cfg.bins:.1f}" for i in x_ticks]
        y_labels = [f"{i*cfg.resp_ref/cfg.bins:.3f}" for i in y_ticks]
        
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_labels)
        ax1.set_xlabel("Densidad final ρ", fontsize=10)
        ax1.set_ylabel(f"Respuesta Δ (ref={cfg.resp_ref})", fontsize=10)
        
        # Delimitadores de celdas alineados a píxeles (sin activar grid estándar para evitar doble grilla)
        for i in range(cfg.bins + 1):
            ax1.axhline(i - 0.5, color='white', linewidth=0.3, alpha=0.6)
            ax1.axvline(i - 0.5, color='white', linewidth=0.3, alpha=0.6)
        
        # Colorbar más informativo
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label("Score-QD", fontsize=9)
        
        # Marcadores mejorados
        try:
            xi_c, yj_c = int(ijk_comp[0]), int(ijk_comp[1])
            xi_q, yj_q = int(best_ijk_qd[0]), int(best_ijk_qd[1])
            xi_e, yj_e = int(best_eq_ijk[0]), int(best_eq_ijk[1])
            
            ax1.scatter([xi_c], [yj_c], s=150, marker='*', c='yellow', 
                        edgecolor='black', linewidths=2, label=f'Mejor cómputo ({desc_comp[3]:.3f})', zorder=10)
            ax1.scatter([xi_q], [yj_q], s=100, marker='o', facecolors='none', 
                        edgecolors='white', linewidths=2, label='Mejor score-QD', zorder=9)
            ax1.scatter([xi_e], [yj_e], s=80, marker='s', c='lime', 
                        alpha=0.8, edgecolor='black', linewidths=1, label='Equilibrado', zorder=8)

            ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
        except Exception:
            pass
        
        # Panel derecho: Mapa de cobertura (celdas ocupadas)
        ax2 = plt.subplot(1, 2, 2)
        
        # Crear mapa de cobertura: 1 donde hay élite, 0 donde no
        cov_map = np.zeros((cfg.bins, cfg.bins))
        for (i, j, k) in elites.keys():
            cov_map[i, j] = 1
        
        im2 = ax2.imshow(cov_map.T, origin='lower', cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
        
        ax2.set_title(f"Cobertura QD\n{cobertura}/{cfg.bins*cfg.bins} celdas ({100*cobertura/(cfg.bins*cfg.bins):.1f}%)", 
                      fontsize=11, pad=10)
        
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels)
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_labels)
        ax2.set_xlabel("Densidad final ρ", fontsize=10)
        ax2.set_ylabel(f"Respuesta Δ (ref={cfg.resp_ref})", fontsize=10)
        
        # Delimitadores de celdas alineados a píxeles (sin activar grid estándar para evitar doble grilla)
        for i in range(cfg.bins + 1):
            ax2.axhline(i - 0.5, color='white', linewidth=0.3, alpha=0.7)
            ax2.axvline(i - 0.5, color='white', linewidth=0.3, alpha=0.7)
        
        # Colorbar binario
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=[0, 1])
        cbar2.set_ticklabels(['Vacío', 'Élite'])
        cbar2.set_label("Ocupación", fontsize=9)
        
        # Mismos marcadores para coherencia
        try:
            ax2.scatter([xi_c], [yj_c], s=150, marker='*', c='yellow', 
                        edgecolor='black', linewidths=2, zorder=10)
            ax2.scatter([xi_q], [yj_q], s=100, marker='o', facecolors='none', 
                        edgecolors='black', linewidths=2, zorder=9)
            ax2.scatter([xi_e], [yj_e], s=80, marker='s', c='lime', 
                        alpha=0.8, edgecolor='black', linewidths=1, zorder=8)
        except Exception:
            pass
        
        plt.tight_layout()
        # Guardar con nombre tradicional esperado por el usuario
        plt.savefig("S1_mapa_QD.png", dpi=160, bbox_inches='tight')
        plt.close()

        # Curva de cobertura MEJORADA
        if coverage_hist:
            plt.figure(figsize=(8, 4))
            
            # Convertir a pasos reales (log_every * iteraciones)
            x_pasos = np.arange(len(coverage_hist)) * cfg.log_every
            cobertura_pct = np.array(coverage_hist) / (cfg.bins * cfg.bins) * 100
            
            plt.subplot(1, 2, 1)
            # Gráfica principal con mejores elementos visuales
            plt.plot(x_pasos, coverage_hist, 'b-', linewidth=2, alpha=0.8, label='Celdas ocupadas')
            
            # Hitos importantes (cada 25% de cobertura)
            max_celdas = cfg.bins * cfg.bins
            hitos = [max_celdas * 0.25, max_celdas * 0.5, max_celdas * 0.75, max_celdas * 0.9]
            hito_nombres = ['25%', '50%', '75%', '90%']
            
            for hito, nombre in zip(hitos, hito_nombres):
                if max(coverage_hist) >= hito:
                    # Encontrar primer momento que se alcanza
                    idx = next((i for i, v in enumerate(coverage_hist) if v >= hito), None)
                    if idx is not None:
                        plt.axhline(hito, color='red', linestyle=':', alpha=0.5)
                        plt.axvline(x_pasos[idx], color='red', linestyle=':', alpha=0.5)
                        plt.text(x_pasos[idx], hito, f' {nombre}\n (it={x_pasos[idx]})', 
                               fontsize=8, va='bottom', ha='left', 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
            
            plt.xlabel("Iteraciones QD", fontsize=10)
            plt.ylabel("Celdas con élite", fontsize=10)
            plt.title(f"Cobertura QD Evolution\n(final: {cobertura}/{max_celdas} = {100*cobertura/max_celdas:.1f}%)", fontsize=11)
            plt.grid(True, alpha=0.4)
            plt.legend(fontsize=9)
            
            # Panel derecho: velocidad de cobertura (derivada)
            plt.subplot(1, 2, 2)
            if len(coverage_hist) > 1:
                velocidad = np.diff(coverage_hist)
                x_vel = x_pasos[1:]
                plt.plot(x_vel, velocidad, 'g-', linewidth=1.5, alpha=0.7, label='Velocidad')
                plt.fill_between(x_vel, velocidad, alpha=0.3, color='green')
                
                # Media móvil de velocidad (usando numpy)
                if len(velocidad) > 10:
                    window = min(10, len(velocidad)//3)
                    vel_smooth = np.convolve(velocidad, np.ones(window)/window, mode='same')
                    plt.plot(x_vel, vel_smooth, 'r-', linewidth=2, label='Tendencia')
                
                plt.xlabel("Iteraciones QD", fontsize=10)
                plt.ylabel("Δ celdas / Δ iteración", fontsize=10)
                plt.title("Velocidad de exploración", fontsize=11)
                plt.grid(True, alpha=0.4)
                plt.legend(fontsize=9)
                
                # Estadísticas de velocidad
                vel_mean = np.mean(velocidad)
                vel_std = np.std(velocidad)
                plt.text(0.02, 0.98, f'μ={vel_mean:.2f}\nσ={vel_std:.2f}', 
                        transform=plt.gca().transAxes, va='top', ha='left', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        fontsize=8)
            
            plt.tight_layout()
            plt.savefig("S1_cobertura.png", dpi=160, bbox_inches='tight')
            plt.close()

        # Figura: evolución de masa en 8 tiempos MEJORADA
        T = hist.shape[0]
        t_poke = cfg.warmup if cfg.warmup > 0 else T//2
        # 8 snapshots: inicio, pre-poke gradual, poke, post-poke gradual, final
        pasos = [
            max(1, T//10),           # muy inicial
            max(1, T//4),            # temprano  
            max(1, t_poke-3),        # pre-poke lejano
            max(1, t_poke-1),        # pre-poke cercano
            min(T-1, t_poke+1),      # post-poke inmediato
            min(T-1, t_poke+5),      # post-poke medio
            min(T-1, T*3//4),        # tardío
            T-1                      # final
        ]
        
        plt.figure(figsize=(16, 4.5))
        
        # Panel superior: evolución temporal con 8 snapshots
        for k, t in enumerate(pasos, 1):
            plt.subplot(2, 8, k)
            # Usar colormap claro y limpio como antes
            plt.imshow(hist[t], cmap='gray', vmin=0, vmax=1)
            
            # Títulos más informativos y GRANDES pero con colores más suaves
            if k == 1:
                plt.title(f"Inicial\nt={t}", fontsize=13, fontweight='bold')
            elif k == 2:
                plt.title(f"Temprano\nt={t}", fontsize=13, color='blue')
            elif k == 3:
                plt.title(f"Pre-poke\nt={t}", fontsize=13, color='blue')
            elif k == 4:
                plt.title(f"Pre-poke\nt={t}", fontsize=13, color='blue', fontweight='bold')
            elif k == 5:
                plt.title(f"Post-poke\nt={t}", fontsize=13, color='red', fontweight='bold')
            elif k == 6:
                plt.title(f"Post-poke\nt={t}", fontsize=13, color='red')
            elif k == 7:
                plt.title(f"Tardío\nt={t}", fontsize=13, color='green')
            else:  # k == 8
                plt.title(f"Final\nt={t}", fontsize=13, color='green', fontweight='bold')
            
            plt.axis('off')
            
            # Agregar colorbar solo en el último
            if k == 8:
                cbar = plt.colorbar(shrink=0.8)
                cbar.set_label('Masa', fontsize=11)
        
        # Panel inferior: gráfica temporal de densidad media
        plt.subplot(2, 1, 2)
        dens_temporal = hist.mean(axis=(1,2))
        plt.plot(dens_temporal, 'b-', linewidth=1.8, alpha=0.8)
        
        # Marcar el poke
        plt.axvline(t_poke, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Poke (t={t_poke})')
        
        # Marcar los tiempos de los 8 snapshots
        colors = ['purple', 'blue', 'blue', 'blue', 'red', 'red', 'green', 'green']
        alphas = [0.7, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7]  # destacar momentos clave
        for i, t in enumerate(pasos):
            plt.axvline(t, color=colors[i], linestyle=':', alpha=alphas[i], linewidth=1.5)
            # Texto más grande y mejor posicionado
            plt.text(t, plt.ylim()[1] * (0.95 - (i % 2) * 0.15), f't={t}', rotation=90, 
                    fontsize=10, ha='right', va='top', color=colors[i], fontweight='bold')
        
        plt.xlabel('Tiempo (pasos)', fontsize=10)
        plt.ylabel('Densidad media', fontsize=10)
        plt.title('Evolución temporal de la densidad', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # Información adicional en el título principal
        plt.suptitle(f"Evolución Autómata Celular S1 (dens≈{desc_comp[0]:.3f}, resp≈{desc_comp[1]:.3f}, comp≈{desc_comp[3]:.3f})", 
                     fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig("S1_evolucion.png", dpi=160, bbox_inches='tight')
        plt.close()

        # Guardar hist del mejor élite (por cómputo) si se solicita
        if args.save_hist:
            try:
                np.save(args.save_hist, hist)
                print(f"Historial guardado en {args.save_hist}")
            except Exception as e:
                print(f"No se pudo guardar hist: {e}")

        # Animación opcional MEJORADA
        if args.anim:
            try:
                render_anim(hist, out_path=args.anim, fps=max(1, int(args.anim_fps)), 
                           cfg=cfg, desc=desc_comp)
                print(f"Animación mejorada guardada en {args.anim}")
            except Exception as e:
                print(f"No se pudo guardar animación: {e}")

        # Top-K por cómputo (desc[3])
        if args.topk_comp and elites:
            # Reutiliza 'pares' ya evaluados arriba (desc consistente)
            pares.sort(key=lambda t: t[2][3], reverse=True)
            topk = pares[:args.topk_comp]
            print("Top-K por cómputo:")
            for rank, (ijk, (p,s), d) in enumerate(topk, 1):
                print(f"  #{rank} celda={ijk} desc=(ρ={d[0]:.3f}, Δ={d[1]:.4f}, σ={d[2]:.4f}, comp={d[3]:.3f})")
            if args.save_topk:
                try:
                    arr_p = np.stack([ps[0] for _, ps, _ in topk], axis=0)
                    arr_s = np.stack([ps[1] for _, ps, _ in topk], axis=0)
                    arr_d = np.stack([d for *_, d in topk], axis=0)
                    np.savez(args.save_topk, pesos=arr_p, sesgo=arr_s, desc=arr_d, cfg=cfg.__dict__)
                    print(f"Top-K guardado en {args.save_topk}")
                except Exception as e:
                    print(f"No se pudo guardar top-K: {e}")
        else:
            print("Sin élites viables (ajusta iteraciones o escalas)")

    # Exportar solo resumen de élites (opcional, solo para debug)
        if cobertura > 0:
            print(f"QD completado: {cobertura} élites encontrados, mejor comp={desc_comp[3]:.4f}")
        else:
            print("QD sin élites - ajusta iteraciones o parámetros")

if __name__ == "__main__":
    demo()
