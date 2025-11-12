# =========================
# FIX 1: Soft wrap
# =========================
import re

def _soft_wrap(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ch in ['_', '/', '\\', '-', '—', ':', '|', '.', ',']:
        text = text.replace(ch, ch + '\u200b')
    def _breaker(m: re.Match) -> str:
        s = m.group(0)
        return '\u200b'.join(s[i:i+20] for i in range(0, len(s), 20))
    text = re.sub(r'\S{40,}', _breaker, text)
    return text

def _pdfreport_mc(self, txt: str, h: float = 6, align: str = "L"):
    self.set_x(self.l_margin)
    w_eff = self.w - self.l_margin - self.r_margin
    self.multi_cell(w_eff, h, _soft_wrap(txt), align=align)
# =========================
# PDFReport — 1 логотип у хедері (справа вгорі, 28 мм, підняте), вирівняний титул і секції
# + ВІДОБРАЖЕННЯ ЗОВНІШНІХ ФАКТОРІВ у титулі та окремим блоком (якщо є в session_state)
# + Фікси: точний 95% PI для Пуассона; фолбек для абсолютних випадків у сценаріях
# =========================
from fpdf import FPDF
from pathlib import Path
import math
import pandas as pd
import streamlit as st  # ⬅️ для читання multiplier / external_factors_selected

try:
    from PIL import Image
except Exception:
    Image = None

class PDFReport(FPDF):
    def __init__(
        self,
        title,
        region,
        district,
        hromada,
        period,
        start_year,
        logo_path=None
    ):
        super().__init__(orientation="P", unit="mm", format="A4")
        # поля та автоперенесення
        self.set_margins(15, 18, 15)
        self.set_auto_page_break(auto=True, margin=15)

        # метадані
        self.title      = str(title)
        self.region     = str(region)
        self.district   = str(district)
        self.hromada    = str(hromada)
        self.period     = str(period)
        self.start_year = str(start_year)
        self.logo_path  = logo_path

        # шрифти (DejaVu → Arial)
        base_fonts = Path(__file__).resolve().parent / "fonts"
        try:
            self.add_font("DejaVu", "", str(base_fonts / "DejaVuSans.ttf"), uni=True)
            self.add_font("DejaVu", "B", str(base_fonts / "DejaVuSans-Bold.ttf"), uni=True)
            self.add_font("DejaVu", "I", str(base_fonts / "DejaVuSans-Oblique.ttf"), uni=True)
            self._font = "DejaVu"
        except Exception:
            self._font = "Arial"
        self.set_font(self._font, "", 12)

        self.alias_nb_pages()
        self.add_page()

    # ---------- утиліти
    def _content_width(self):
        return self.w - self.l_margin - self.r_margin

    def _hr(self, pad=3):
        self.ln(pad)
        self.set_draw_color(180, 180, 220)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(pad)

    # санітизація населення з рядка/числа
    def _clean_pop(self, x):
        s = str(x).replace('\u00a0','').replace(' ', '').replace(',', '')
        try:
            return int(float(s))
        except Exception:
            return 0

    # Точний 95% предиктивний інтервал для X ~ Poisson(lam) без SciPy
    def _poisson_pi_counts(self, lam: float, alpha: float = 0.05):
        lam = max(float(lam), 0.0)
        if lam == 0.0:
            return 0, 0
        p = math.exp(-lam)   # P(X=0)
        cdf = p              # F(0)
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        # нижня межа
        k = 0
        if cdf >= lower_q:
            L = 0
        else:
            while cdf < lower_q:
                k += 1
                p = p * lam / k
                cdf += p
            L = k

        # верхня межа
        while cdf < upper_q:
            k += 1
            p = p * lam / k
            cdf += p
        U = k
        return int(L), int(U)

    # ---------- Header/Footer (1 логотип НАД лінією)
    def header(self):
        logo_w_right = 28
        y_logo_top   = 2   # вище, щоб не «сідав» на лінію
        gap_below    = 4   # відступ під лого

        logo_h = 0
        if self.logo_path and Path(self.logo_path).exists():
            try:
                if Image is not None:
                    with Image.open(self.logo_path) as im:
                        w_px, h_px = im.size
                        if w_px > 0:
                            logo_h = logo_w_right * (h_px / float(w_px))
                if not logo_h:
                    logo_h = logo_w_right
                x_logo = self.w - self.r_margin - logo_w_right
                self.image(self.logo_path, x=x_logo, y=y_logo_top, w=logo_w_right)
            except Exception:
                pass

        # Лінія під логотипом
        y_line = y_logo_top + logo_h + gap_below
        self.set_draw_color(200, 210, 255)
        self.set_line_width(0.6)
        self.line(self.l_margin, y_line, self.w - self.r_margin, y_line)

        # Курсор для контенту
        self.set_xy(self.l_margin, y_line + 6)

    def footer(self):
        self.set_y(-12)
        self.set_font(self._font, "", 10)
        self.cell(0, 8, f"Сторінка {self.page_no()}/{{nb}}", align="C")

    # ---------- ТИТУЛЬНИЙ БЛОК (додає короткий рядок про зовнішні фактори, якщо вони застосовані)
    def add_title_block(self, dt_str: str, meta: dict | None = None):
        meta = meta or {}
        region   = str(meta.get("region", self.region))
        district = str(meta.get("district", self.district))
        hromada  = str(meta.get("hromada", self.hromada))
        period   = str(meta.get("period", self.period))
        start    = str(meta.get("start_year", self.start_year))
        horizon  = meta.get("horizon", None)

        # старт нижче лінії хедера
        self.set_y(max(self.get_y(), 40))

        # Заголовок
        self.set_font(self._font, "B", 22)
        self.multi_cell(self._content_width(), 12, self.title, align="C")

        # Дата і час
        self.set_font(self._font, "", 11)
        self.set_x(self.l_margin)
        self.multi_cell(self._content_width(), 6, f"Дата і час формування: {dt_str}", align="C")

        # ⬇️ Короткий рядок про зовнішні фактори (якщо був застосований множник)
        try:
            m_ext = float(st.session_state.get("ext_factor_multiplier", 1.0))
            total_ext = st.session_state.get("__ext_total_pct__")
            if total_ext is None:
                rows = st.session_state.get("external_factors_selected") or []
                total_ext = float(sum(float(r.get("Вплив (%)", 0)) for r in rows)) if rows else 0.0
            if m_ext and abs(m_ext - 1.0) > 1e-9 and total_ext is not None:
                self.set_font(self._font, "I", 10)
                self.set_text_color(80, 80, 80)
                self.multi_cell(
                    self._content_width(), 5,
                    f"Зовнішні фактори застосовано: +{float(total_ext):.0f}% (множник ×{m_ext:.2f})",
                    align="C"
                )
                self.set_text_color(0, 0, 0)
        except Exception:
            pass

        self.ln(2)

        # Метадані
        self.set_font(self._font, "", 12)
        rows = [
            f"Область: {region}",
            f"Район: {district}",
            f"Громада: {hromada}",
            f"Період прогнозу: {period}   Рік початку: {start}",
        ]
        if horizon is not None:
            rows.append(f"Тривалість прогнозу: {horizon} років")

        for s in rows:
            self.set_x(self.l_margin)
            self.multi_cell(self._content_width(), 7, s, align="L")

        self._hr(6)

    # ---------- ДОДАТКОВО: повний блок «Зовнішні фактори» (таблиця)
    def add_external_factors_block(self, rows_like=None):
        """
        Відобразити обрані зовнішні фактори у PDF.
        rows_like: None | list[dict] | pd.DataFrame — очікує колонки «Фактор», «Вплив (%)»
        Якщо None — намагається прочитати з st.session_state["external_factors_selected"].
        """
        try:
            if rows_like is None:
                rows_like = st.session_state.get("external_factors_selected", [])
            if isinstance(rows_like, list):
                df_ext = pd.DataFrame(rows_like)
            elif isinstance(rows_like, pd.DataFrame):
                df_ext = rows_like.copy()
            else:
                return  # нічого показувати

            # валідація
            if df_ext.empty or "Фактор" not in df_ext.columns or "Вплив (%)" not in df_ext.columns:
                return

            # агрегати
            total_impact = float(df_ext["Вплив (%)"].fillna(0).astype(float).sum())
            try:
                m_ext = float(st.session_state.get("ext_factor_multiplier", 1.0))
            except Exception:
                m_ext = 1.0

            # заголовок
            self.set_font(self._font, "B", 14)
            self.cell(0, 8, "Зовнішні фактори (застосовано до прогнозу)", ln=1)

            # таблиця
            show_cols = ["Фактор", "Вплив (%)"]
            if "Діапазон (рек.)" in df_ext.columns:
                show_cols.append("Діапазон (рек.)")
            if "Примітка" in df_ext.columns:
                show_cols.append("Примітка")

            # шапка
            cw = self._content_width()
            if len(show_cols) == 2:
                widths = [cw * 0.65, cw * 0.35]
            elif len(show_cols) == 3:
                widths = [cw * 0.50, cw * 0.20, cw * 0.30]
            else:
                widths = [cw * 0.45, cw * 0.18, cw * 0.17, cw * 0.20]

            self.set_fill_color(230, 240, 255)
            self.set_font(self._font, "B", 11)
            for i, h in enumerate(show_cols):
                self.cell(widths[i], 8, str(h), 1, 0, "C", True)
            self.ln()

            self.set_font(self._font, "", 11)
            for _, r in df_ext.iterrows():
                for i, c in enumerate(show_cols):
                    v = r.get(c, "")
                    self.cell(widths[i], 8, str(v), 1, 0, "C")
                self.ln()

            # підсумок
            self.ln(2)
            self.set_font(self._font, "", 11)
            self.multi_cell(
                0, 6,
                f"Сумарний номінальний вплив: {total_impact:.0f}%  •  Застосований множник: ×{m_ext:.2f}"
            )
            self._hr(3)
        except Exception:
            # тихо ігноруємо, щоб не ламати PDF
            pass

    # ---------- базова таблиця
    def _table(self, df, cols, col_widths=None, header_fill=(230, 240, 255), align="C"):
        cw = self._content_width()
        n = len(cols)
        if not col_widths:
            w0 = 25
            rest = max(cw - w0, 1)
            col_widths = [w0] + [rest / (n - 1)] * (n - 1)
        s = sum(col_widths[:-1])
        col_widths[-1] = max(cw - s, 1)

        self.set_fill_color(*header_fill)
        self.set_font(self._font, "B", 11)
        for i, c in enumerate(cols):
            self.cell(col_widths[i], 8, str(c), 1, 0, align, True)
        self.ln()

        self.set_font(self._font, "", 11)
        for _, r in df.iterrows():
            for i, c in enumerate(cols):
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.1f}"
                self.cell(col_widths[i], 8, str(v), 1, 0, align)
            self.ln()

    # ---------- сценарне прогнозування (повний блок) + фолбек на випадок 1–2 у всіх роках
    def add_scenario_table(self, df_inc: pd.DataFrame, df_abs: pd.DataFrame):
        self.set_font(self._font, "B", 14)
        self.cell(0, 8, "Сценарне прогнозування", ln=1)

        # Інцидентність
        self.ln(1)
        self.set_font(self._font, "B", 12)
        self.cell(0, 6, "Інцидентність (на 100 тис.)", ln=1)
        self._table(df_inc, ["Рік", "Оптимістичний", "Середній", "Песимістичний"])

        # --- м'який фолбек, якщо df_abs підозрілий і є населення ---
        try:
            # визначаємо назви колонок
            if {"Опт","Сер","Пес"}.issubset(df_abs.columns):
                to_display = df_abs.copy()
                to_display["Оптимістичний"] = to_display["Опт"]
                to_display["Середній"]      = to_display["Сер"]
                to_display["Песимістичний"] = to_display["Пес"]
                to_display = to_display[["Рік","Оптимістичний","Середній","Песимістичний"]]
            else:
                to_display = df_abs.copy()[["Рік","Оптимістичний","Середній","Песимістичний"]]

            vals = pd.to_numeric(
                to_display[["Оптимістичний","Середній","Песимістичний"]].values.ravel(),
                errors="coerce"
            )
            suspicious = pd.notna(vals).all() and (pd.Series(vals).max() <= 2)

            if suspicious:
                # 🔧 ВАЖЛИВО: беремо effective_population як пріоритет
                N_raw = st.session_state.get("effective_population", st.session_state.get("population", None))
                N = self._clean_pop(N_raw) if N_raw is not None else 0
                if N > 0:
                    rec = (df_inc[["Оптимістичний","Середній","Песимістичний"]] * N / 100000.0).round().astype(int)
                    to_display = pd.DataFrame({
                        "Рік": df_inc["Рік"].astype(int),
                        "Оптимістичний": rec["Оптимістичний"],
                        "Середній":      rec["Середній"],
                        "Песимістичний": rec["Песимістичний"]
                    })
        except Exception:
            to_display = df_abs if "df_abs" in locals() else pd.DataFrame()

        # Абсолютні
        self.ln(2)
        self.set_font(self._font, "B", 12)
        self.cell(0, 6, "Абсолютні випадки", ln=1)
        self._table(to_display, ["Рік","Оптимістичний","Середній","Песимістичний"])

        # --- Безпечне форматування чисел для комбінованої таблиці ---
        def _int_str(x):
            s = str(x).strip()
            if s in {"", "-", "nan", "None", "none", "NaN"}:
                return "-"
            try:
                s2 = s.replace("\u00a0","").replace(" ","").replace(",","")
                return f"{int(float(s2))}"
            except Exception:
                try:
                    return f"{int(x)}"
                except Exception:
                    return s  # як є

        # Комбінована
        self.ln(2)
        self.set_font(self._font, "B", 12)
        self.cell(0, 6, "Комбінована (інцидентність / випадки)", ln=1)
        inc_combo = (
            df_inc["Оптимістичний"].map(lambda x: f"{float(x):.1f}") + " / " +
            df_inc["Середній"].map(lambda x: f"{float(x):.1f}") + " / " +
            df_inc["Песимістичний"].map(lambda x: f"{float(x):.1f}")
        )
        cases_combo = (
            to_display["Оптимістичний"].map(_int_str) + " / " +
            to_display["Середній"].map(_int_str)      + " / " +
            to_display["Песимістичний"].map(_int_str)
        )
        df_combo = pd.DataFrame({"Рік": df_inc["Рік"], "Інц.": inc_combo, "Випадки": cases_combo})
        self._table(df_combo, ["Рік","Інц.","Випадки"], col_widths=[25, 80, 75])
        self._hr(3)

    # ---------- графік сценарного прогнозування
    def add_scenario_chart(self, chart_path):
        if chart_path and Path(chart_path).exists():
            self.set_font(self._font, "B", 14)
            self.cell(0, 8, "Графік сценарного прогнозування", ln=1)
            self.image(chart_path, x=self.l_margin, w=self._content_width())
            self._hr(3)

    # ---------- Пуассон (точний інтервал)
    def add_poisson_blocks(self, df_inc_mid: pd.DataFrame, population: int, alpha: float = 0.05):
        N = max(int(population or 0), 0)
        self.set_font(self._font, "B", 14)
        self.cell(0, 8, "Прогноз за методом Пуассона", ln=1)
        self.set_font(self._font, "", 11)

        if N <= 0:
            self.multi_cell(0, 6, "Неможливо розрахувати без населення N > 0.")
            self._hr(3)
            return

        # Абсолютні випадки (середній)
        self.ln(2)
        self.set_font(self._font, "B", 12)
        self.cell(0, 7, "Абсолютні випадки (середній): очікуване та 95% ДІ", ln=1)

        headers = ["Рік", "Очікуване", "Нижня 95% ДІ", "Верхня 95% ДІ"]
        w = [25, 40, 55, 55]

        self.set_fill_color(230, 240, 255)
        self.set_font(self._font, "B", 11)
        for i, h in enumerate(headers):
            self.cell(w[i], 8, h, 1, 0, "C", True)
        self.ln()

        self.set_font(self._font, "", 11)
        rows_cases = []
        for _, r in df_inc_mid.iterrows():
            y = int(r["Рік"])
            mid_inc = float(r["Середній"])
            lam = mid_inc * N / 100000.0            # очікувана кількість випадків (float)
            L, U = self._poisson_pi_counts(lam, alpha=alpha)
            rows_cases.append([y, lam, L, U])
            self.cell(w[0], 8, str(y), 1, 0, "C")
            self.cell(w[1], 8, f"{lam:.0f}", 1, 0, "C")
            self.cell(w[2], 8, f"{L:d}", 1, 0, "C")
            self.cell(w[3], 8, f"{U:d}", 1, 1, "C")

        # Інцидентність
        self.ln(2)
        self.set_font(self._font, "B", 12)
        self.cell(0, 7, "Інцидентність (середній): очікуване та 95% ДІ (на 100 тис.)", ln=1)

        self.set_font(self._font, "B", 11)
        for i, h in enumerate(headers):
            self.cell(w[i], 8, h, 1, 0, "C", True)
        self.ln()

        self.set_font(self._font, "", 11)
        for y, lam, L, U in rows_cases:
            i_mid = lam * 100000.0 / N
            i_lo  = L   * 100000.0 / N
            i_hi  = U   * 100000.0 / N
            self.cell(w[0], 8, str(y), 1, 0, "C")
            self.cell(w[1], 8, f"{i_mid:.1f}", 1, 0, "C")
            self.cell(w[2], 8, f"{i_lo:.1f}", 1, 0, "C")
            self.cell(w[3], 8, f"{i_hi:.1f}", 1, 1, "C")
        self._hr(3)

    # ---------- t-Стьюдента
    def _t_crit_95(self, df:int):
        table = {
            1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
            11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
            21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042
        }
        return table.get(max(1, min(df, 30)), 1.96)

    def add_student_block(self, df_hist: pd.DataFrame, alpha: float = 0.05):
        vals = []
        try:
            for v in df_hist["Захворюваність"]:
                vals.append(float(v))
        except Exception:
            pass

        self.set_font(self._font, "B", 14)
        self.cell(0, 8, "Прогноз за методом Стьюдента", ln=1)

        if len(vals) < 2:
            self.set_font(self._font, "", 11)
            self.multi_cell(0, 6, "Недостатньо даних для оцінки (потрібно ≥ 2 спостереження).")
            self._hr(3)
            return

        n = len(vals)
        xbar = sum(vals)/n
        ss = sum((x - xbar)**2 for x in vals)
        s  = (ss / (n-1))**0.5
        se = s / (n**0.5)
        tcrit = self._t_crit_95(n-1)
        ci = (xbar - tcrit*se, xbar + tcrit*se)
        pi = (xbar - tcrit*s*(1 + 1/n)**0.5, xbar + tcrit*s*(1 + 1/n)**0.5)

        self.set_font(self._font, "", 11)
        self.multi_cell(
            0, 6,
            "Використовуємо t-розподіл Стьюдента для оцінки невизначеності.\n"
            " • 95% ДІ середнього:  x̄ ± t_{0.975, n−1} · s/√n\n"
            " • 95% інтервал прогнозу:  x̄ ± t_{0.975, n−1} · s · √(1+1/n)"
        )
        self.ln(2)

        df_stats = pd.DataFrame([
            ["n", n],
            ["x̄", f"{xbar:.2f}"],
            ["s", f"{s:.2f}"],
            ["SE", f"{se:.3f}"],
            ["t (df="+str(n-1)+")", f"{tcrit:.3f}"],
            ["95% ДІ, низ", f"{ci[0]:.2f}"],
            ["95% ДІ, верх", f"{ci[1]:.2f}"],
            ["95% PI, низ", f"{pi[0]:.2f}"],
            ["95% PI, верх", f"{pi[1]:.2f}"],
        ], columns=["Показник","Значення"])
        cw = [60, self._content_width()-60]
        self.set_fill_color(230,240,255)
        self.set_font(self._font, "B", 11)
        self.cell(cw[0], 8, "Показник", 1, 0, "C", True)
        self.cell(cw[1], 8, "Значення", 1, 1, "C", True)
        self.set_font(self._font, "", 11)
        for _, r in df_stats.iterrows():
            self.cell(cw[0], 8, str(r["Показник"]), 1, 0, "C")
            self.cell(cw[1], 8, str(r["Значення"]), 1, 1, "C")
        self._hr(3)

    # ---------- висновки
    def add_conclusions(self, text=None):
        self.ln(2)
        self.set_font(self._font, "B", 12)
        self.cell(0, 7, "Висновки", ln=1)
        self.set_font(self._font, "", 11)
        default_text = (
            "Сформовані сценарії демонструють можливі траєкторії захворюваності та "
            "навантаження на систему охорони здоров’я. Оцінки невизначеності "
            "допомагають планувати ресурси."
        )
        self.multi_cell(0, 6, text or default_text)

    # ---------- порівняння (2 графіки на одній сторінці)
    def add_comparison_page(self, war_png=None, nowar_png=None,
                            title_war="З воєнними роками",
                            title_nowar="Без воєнних років"):
        self.add_page()
        self.set_font(self._font, "B", 14)
        self.cell(0, 8, "Порівняння прогнозів (врахування воєнних років)", ln=1)

        left = self.l_margin
        top_y = self.get_y() + 2
        usable_w = self._content_width()
        usable_h = self.h - top_y - self.b_margin

        caption_h = 6.0
        gap = 4.0
        chart_h = max(40.0, (usable_h - caption_h - gap - caption_h) / 2.0)

        # верхній
        self.set_font(self._font, "", 11)
        self.set_xy(left, top_y)
        self.cell(0, caption_h, title_war, ln=1)
        y1 = self.get_y()
        if war_png and Path(war_png).exists():
            self.image(war_png, x=left, y=y1, w=usable_w, h=chart_h)
        self.set_y(y1 + chart_h)

        # проміжок
        self.ln(gap)

        # нижній
        self.set_font(self._font, "", 11)
        self.cell(0, caption_h, title_nowar, ln=1)
        y2 = self.get_y()
        if nowar_png and Path(nowar_png).exists():
            self.image(nowar_png, x=left, y=y2, w=usable_w, h=chart_h)
        self.set_y(y2 + chart_h)

# =========================
# STABILITY HELPERS
# =========================
def _safe_int(x, default=0):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return int(x)
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            if s == "": return default
            return int(float(s))
    except Exception:
        return default

def _safe_float(x, default=0.0):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            if s == "": return default
            return float(s)
    except Exception:
        return default

def _norm_period(p: str) -> str:
    m = str(p or "").strip().lower()
    mapping = {
        "covid-19": "COVID-19",
        "covid": "COVID-19",
        "початок війни": "початок війни",
        "повномасштабне вторгнення": "повномасштабне вторгнення",
        "до повномасштабного": "до повномасштабного вторгнення",
        "мирний час": "мирний час",
        "післявоєнний": "післявоєнний",
    }
    return mapping.get(m, p if p else "")

def _auto_period_for_year(y: int) -> str | None:
    y = _safe_int(y, 0)
    if 2020 <= y <= 2021: return "COVID-19"
    if y == 2022: return "початок війни"
    if 2023 <= y <= 2025: return "повномасштабне вторгнення"
    if y < 2020: return "до повномасштабного вторгнення"
    return None

def _clean_incidence_rows(rows):
    out = []
    for r in rows or []:
        try:
            y = _safe_int(r.get("Рік"), None)
            inc = _safe_float(r.get("Захворюваність"), None)
            per = str(r.get("Період", "")).strip()
            if y is None or inc is None: continue
            if inc < 0: inc = 0.0
            if per == "": per = _auto_period_for_year(y)
            out.append({"Рік": int(y), "Захворюваність": float(inc), "Період": _norm_period(per)})
        except Exception:
            continue
    # dedup
    cleaned, used = [], set()
    for r in reversed(out):
        if r["Рік"] in used: continue
        used.add(r["Рік"]); cleaned.append(r)
    cleaned.reverse()
    return cleaned

def _get_or_build_incidence_df():
    import pandas as pd
    rows = _clean_incidence_rows(st.session_state.get("incidence_data", []))
    df = pd.DataFrame(rows, columns=["Рік","Захворюваність","Період"]) if rows else pd.DataFrame(columns=["Рік","Захворюваність","Період"])
    if not df.empty:
        df["Рік"] = df["Рік"].apply(_safe_int)
        df["Захворюваність"] = df["Захворюваність"].apply(lambda v: max(0.0, _safe_float(v)))
        df = df.replace([math.inf, -math.inf], float("nan")).dropna(subset=["Рік","Захворюваність"]).sort_values("Рік").reset_index(drop=True)
    return df

def safe_session_defaults():
    for k, v in {
        "incidence_data": [],
        "forecast_period": "повномасштабне вторгнення",
        "_fp_norm": None,
    }.items():
        if k not in st.session_state: st.session_state[k] = v

def get_period_norm():
    safe_session_defaults()
    raw = st.session_state.get("forecast_period", "повномасштабне вторгнення")
    val = _norm_period(raw)
    if st.session_state.get("_fp_norm") != val:
        st.session_state["_fp_norm"] = val
    return st.session_state["_fp_norm"]

# ========================= END STABILITY HELPERS =========================
import sys
import os
import streamlit as st

# =========================
# Unicode-шрифт для PDF (тільки DejaVuSans)
# =========================
import io
from fpdf import FPDF
from PyPDF2 import PdfMerger

FONT_OK = False
FONT_FAMILY = "DejaVu"

def _pdf_load_unicode_fonts(pdf):
    """
    Підключає лише Unicode-шрифт DejaVu (звичайний/жирний/курсив).
    Якщо не знайдено — виводиться попередження.
    """
    global FONT_OK, FONT_FAMILY
    FONT_OK = False
    FONT_FAMILY = "DejaVu"

    for prefix in ("", "fonts/"):
        try:
            pdf.add_font("DejaVu", "", f"{prefix}DejaVuSans.ttf", uni=True)
            pdf.add_font("DejaVu", "B", f"{prefix}DejaVuSans-Bold.ttf", uni=True)
            try:
                pdf.add_font("DejaVu", "I", f"{prefix}DejaVuSans-Oblique.ttf", uni=True)
            except Exception:
                pdf.add_font("DejaVu", "I", f"{prefix}DejaVuSans-Italic.ttf", uni=True)
            FONT_OK = True
            return
        except Exception:
            continue
    FONT_OK = False
    FONT_FAMILY = "DejaVu"

# =========================
# Клас PDF
# =========================
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo_bottom_page = False  # прапорець для розміщення логотипа внизу

    def header(self):
        pass

    def footer(self):
        pass

# =========================
# BOOTSTRAP: guarantee full_df exists
# =========================
try:
    full_df  # noqa: F821
except NameError:
    try:
        full_df = _get_or_build_incidence_df()
    except Exception:
        full_df = None
else:
    try:
        _ = full_df[["Рік", "Захворюваність"]]
    except Exception:
        try:
            full_df = _get_or_build_incidence_df()
        except Exception:
            full_df = None

# =========================
# Інші імпорти
# =========================
import pandas as pd
from pathlib import Path
from io import BytesIO
import base64
from datetime import datetime
import json
import hashlib
from fpdf import FPDF
import os

# =========================
# Допоміжна функція
# =========================
def _normalize_period_column(df):
    if df is None:
        return df
    if "Період" not in df.columns:
        df["Період"] = ""
    else:
        df["Період"] = df["Період"].astype(str).str.strip()
    return df

# =========================
# Функція для ресурсів (PyInstaller)
# =========================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    except Exception:
        pass
    return os.path.join(base_path, relative_path)

# =========================
# Шляхи / активи
# =========================
BASE_DIR = Path(resource_path("."))
ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR  = BASE_DIR / "fonts"
LOGO_PROGRAM = ASSETS_DIR / "logo_program.png"
LOGO_PDF     = ASSETS_DIR / "logo_pdf.png"

# =========================
# Сторінка
# =========================
st.set_page_config(page_title="ТБ-Модель UA", layout="wide")

# === Примусовий СВІТЛИЙ фон застосунку ===
st.markdown("""
<style>
:root { color-scheme: light; }

/* Глобальний білий фон і темний текст */
html, body, .stApp, [data-testid="stAppViewContainer"],
section.main, section.main > div.block-container {
  background: #ffffff !important;
  color: #111111 !important;
}

/* Перекриття темної теми, якщо вона увімкнеться системно */
[data-theme="dark"] {
  color-scheme: light !important;
}

/* Карти/контейнери — світлі */
div[data-testid="stVerticalBlock"] div[tabindex="0"] {
  background: #ffffff !important;
}

/* Елементи вводу — світлі */
input, textarea, select {
  background: #ffffff !important;
  color: #111111 !important;
}

/* Кнопки — темний текст на світлому */
.stButton > button {
  background: #ffffff !important;
  color: #111111 !important;
  border: 1px solid #ced4da !important;
}
.stDownloadButton > button {
  background: #ffffff !important;
  color: #111111 !important;
  border: 1px solid #ced4da !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Глобальний захист від "вильотів" (мінімальний)
# =========================
def safe_session_defaults():
    ss = st.session_state
    def _sf(x, d=0.0):
        try:
            if x is None or (isinstance(x, str) and not x.strip()):
                return d
            return float(x)
        except Exception:
            return d
        except Exception:
            pass
    def _si(x, d=0):
        try:
            if x is None or (isinstance(x, str) and not x.strip()):
                return d
            return int(float(x))
        except Exception:
            return d
        except Exception:
            pass
    if not isinstance(ss.get("forecast_period"), str) or not ss.get("forecast_period", "").strip():
        ss["forecast_period"] = "повномасштабне вторгнення"
    if "duration_years" not in ss:
        ss["duration_years"] = _si(ss.get("duration", 5), 5)
    else:
        ss["duration_years"] = _si(ss.get("duration_years", 5), 5)
safe_session_defaults()

# Скрол у верх на кожний ререндер
st.components.v1.html("<script>window.top.scrollTo(0,0);</script>", height=0)

# CSS: select z-index та стилі кнопки «очистити»
st.markdown(
    """
<style>
.stSelectbox, [data-baseweb="select"] { z-index: 1000; }

/* Кнопка «Очистити всю форму» — прозорий фон, чорна обводка */
.clear-btn > button {
  background: transparent !important;
  color: #000 !important;
  border: 2px solid #000 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: 6px 12px !important;
  box-shadow: none !important;
}
.clear-btn > button:hover { transform: translateY(-1px); }
</style>
""",
    unsafe_allow_html=True,
)

# --- Visual anti-cropping patch ---
st.markdown("""
<style>
:root { --tb-input-radius: 10px; }

/* Selectbox wrapper */
div[data-testid="stSelectbox"] > div {
  border-radius: var(--tb-input-radius) !important;
  overflow: visible !important;
}

/* BaseWeb select control box */
div[data-baseweb="select"] > div {
  border-radius: var(--tb-input-radius) !important;
  overflow: visible !important;
  background: #ffffff !important;
  border: 1px solid #ced4da !important;
}

/* Text input wrapper & input itself */
div[data-testid="stTextInput"] > div {
  border-radius: var(--tb-input-radius) !important;
  overflow: visible !important;
  background: transparent !important;
}
div[data-testid="stTextInput"] input {
  border-radius: var(--tb-input-radius) !important;
  background: #ffffff !important;
  border: 1px solid #ced4da !important;
  height: 44px !important;
  padding: 8px 12px !important;
}

/* Remove extra right padding */
.stSelectbox, .stTextInput { padding-right: 0 !important; }

/* Slightly increase container padding-right */
section.main > div.block-container { padding-right: 1.25rem !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Безпечне автозбереження сесії
# =========================
SESSION_FILE = BASE_DIR / "tb_model_ua_last_session.json"

_BLOCKED_PREFIXES = (
    "FormSubmitter:", "Button", "RadioGroup:", "Checkbox:", "TextInput",
    "Select", "Slider:", "DownloadButton", "FileUploader", "MultiFileUploader",
    "btn_",  # службові
)
def _is_blocked_key(k: str) -> bool:
    if ":" in k:
        return True
    return any(str(k).startswith(p) for p in _BLOCKED_PREFIXES)

def _save_session_safe():
    try:
        session_data = {}
        for k, v in st.session_state.items():
            if _is_blocked_key(k):
                continue
            if isinstance(v, (int, float, str, list, dict)):
                session_data[k] = v
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        try:
            pr_safe_int(f"[!] Не вдалося зберегти сесію: {e}")
        except Exception:
            try:
                st.warning(f"⚠️ Не вдалося зберегти сесію: {e}")
            except Exception:
                pass
            except Exception:
                pass
        except Exception:
            pass

def _load_session():
    try:
        if not SESSION_FILE.exists():
            return
    except Exception:
        pass

        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("_downloaded_pdf", False):
            try:
                SESSION_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            except Exception:
                pass
            return

        for k, v in data.items():
            if _is_blocked_key(k):
                continue
            if k in st.session_state:
                continue
            st.session_state[k] = v

# =========================
# Шапка: лого | назва | кнопка очистити (праворуч)
# =========================
def _clear_all():
    try:
        st.session_state.clear()
    except Exception:
        pass

col_logo, col_title, col_btn = st.columns([0.12, 0.68, 0.20])
with col_logo:
    if LOGO_PROGRAM.exists():
        logo_b64 = base64.b64encode(LOGO_PROGRAM.read_bytes()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" alt="Лого" width="110" style="vertical-align:middle;">',
            unsafe_allow_html=True,
        )
with col_title:
    st.markdown('<h1 style="margin:0;line-height:1;">ТБ-Модель UA</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="margin:0;line-height:1;"><b>Програма сценарного прогнозування туберкульозу</b></p>',
        unsafe_allow_html=True,
    )
with col_btn:
    st.markdown('<div class="clear-btn" style="display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
    if st.button("🧹 Очистити всю форму", key="btn_clear_all", help="Стерти всі введені значення та перезапустити застосунок"):
        _clear_all()
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Утиліти
# =========================
def _fmt__safe_int(n):
    try:
        return f"{_safe_int(round(n)):,}".replace(",", " ")
    except Exception:
        return str(n)
    except Exception:
        pass

def wrap_long_words(text, max_len=50):
    if not isinstance(text, str):
        text = str(text)
    parts, out = text.split(), []
    for p in parts:
        if len(p) > max_len:
            out.extend([p[i:i+max_len] for i in range(0, len(p), max_len)])
        else:
            out.append(p)
    return " ".join(out)

# =========================
# build_pdf_report — з графіками сценарію та Пуассона (fixed)
# =========================
from pathlib import Path
import math, datetime, tempfile
from fpdf import FPDF

class _PDF(FPDF):
    def footer(self):
        nm = getattr(self, "_font_name", "Arial")
        self.set_y(-15)
        self.set_font(nm, "", 8)
        self.cell(0, 10, f"Сторінка {self.page_no()}/{{nb}}", align="C")

def _register_unicode_fonts(pdf: FPDF):
    candidates = [
        ("DejaVu", Path("DejaVuSans.ttf"), Path("DejaVuSans-Bold.ttf")),
        ("Arial", Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    ]
    font_name, bold_avail = None, False
    for name, reg, bold in candidates:
        try:
            if reg.exists():
                pdf.add_font(name, "", str(reg), uni=True)
                font_name = name
                if bold and bold.exists():
                    pdf.add_font(name, "B", str(bold), uni=True)
                    bold_avail = True
                break
        except Exception:
            pass
    if not font_name:
        font_name, bold_avail = "Arial", True
    return font_name, bold_avail

def _epw(pdf: FPDF):
    return pdf.w - pdf.l_margin - pdf.r_margin

def _table(pdf: FPDF, font_name: str, bold_avail: bool, headers, rows, widths=None, fs_head=11, fs_row=10):
    epw = _epw(pdf)
    if widths is None:
        widths = [epw / len(headers)] * len(headers)
    pdf.set_font(font_name, "B" if bold_avail else "", fs_head)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, str(h), border=1, align="C")
    pdf.ln()
    pdf.set_font(font_name, "", fs_row)
    for row in rows:
        for i, v in enumerate(row):
            pdf.cell(widths[i], 6, str(v), border=1, align="C")
        pdf.ln()
    pdf.ln(3)

def build_pdf_report(
    region, district, community, category, fperiod,
    pop_prewar, pop_current, pop_return, pop_postwar,
    start_year, forecast_years,
    opt_A, mid_A, pes_A,
    chart_buf,
    poisson_chart_buf=None,
    ext_rows=None, ext_total_pct=None, ext_multiplier=None
):
    # ---------- helpers (FIX) ----------
    def _clean_pop(x):
        s = str(x).replace('\u00a0','').replace(' ', '').replace(',', '')
        try:
            return int(float(s)) if s not in ('', 'None', 'nan') else 0
        except Exception:
            return 0

    def _poisson_pi_counts(lam: float, alpha: float = 0.05):
        """Точний 95% інтервал для X~Poisson(lam) без SciPy. Повертає (L, U)."""
        lam = max(float(lam), 0.0)
        if lam == 0.0:
            return 0, 0
        p = math.exp(-lam)   # P(X=0)
        cdf = p              # F(0)
        lower_q = alpha/2.0
        upper_q = 1.0 - alpha/2.0
        # нижня межа
        k = 0
        if cdf >= lower_q:
            L = 0
        else:
            while cdf < lower_q:
                k += 1
                p = p * lam / k
                cdf += p
            L = k
        # верхня межа
        while cdf < upper_q:
            k += 1
            p = p * lam / k
            cdf += p
        U = k
        return int(L), int(U)

    pdf = _PDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()

    font_name, bold_avail = _register_unicode_fonts(pdf)
    pdf._font_name = font_name
    pdf._bold_avail = bold_avail

    # ---------- СТОРІНКА 1 ----------
    pdf.add_page()
    epw = _epw(pdf)

    # Титул
    pdf.set_font(font_name, "B" if bold_avail else "", 14)
    pdf.multi_cell(0, 10, "Звіт моделювання туберкульозу (ТБ-Модель UA)", align="C")
    pdf.set_font(font_name, "", 10)
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.cell(0, 6, f"Дата формування звіту: {now}", ln=1)

    # ---------- СЦЕНАРНЕ ПРОГНОЗУВАННЯ ----------
    pdf.ln(2)
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, "Прогноз інцидентності (на 100 тис.)", ln=1)
    headers = ["Рік", "Опт.", "Сер.", "Пес."]
    rows = [[y, f"{opt_A[i]:.1f}", f"{mid_A[i]:.1f}", f"{pes_A[i]:.1f}"] for i, y in enumerate(forecast_years)]
    _table(pdf, font_name, bold_avail, headers, rows)

    # ---------- Абсолютні випадки (FIX: коректне N + без дефолту 1) ----------
    pop_candidates = [_clean_pop(pop_current), _clean_pop(pop_postwar), _clean_pop(pop_return), _clean_pop(pop_prewar)]
    pop_used = next((p for p in pop_candidates if p > 0), 0)

    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, "Прогноз абсолютних випадків", ln=1)

    if pop_used > 0:
        abs_opt = [int(round(opt_A[i] * pop_used / 100000.0)) for i in range(len(forecast_years))]
        abs_mid = [int(round(mid_A[i] * pop_used / 100000.0)) for i in range(len(forecast_years))]
        abs_pes = [int(round(pes_A[i] * pop_used / 100000.0)) for i in range(len(forecast_years))]
        rows2 = [[y, abs_opt[i], abs_mid[i], abs_pes[i]] for i, y in enumerate(forecast_years)]
    else:
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, "Увага: населення N не задане/некоректне — абсолютні випадки не розраховано.")
        rows2 = [[y, "-", "-", "-"] for y in forecast_years]
    _table(pdf, font_name, bold_avail, headers, rows2)

    # Графік сценарного прогнозу
    if chart_buf:
        try:
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            with open(tmp_path, "wb") as f:
                f.write(chart_buf.getvalue() if hasattr(chart_buf, "getvalue") else chart_buf)
            pdf.set_font(font_name, "B", 12)
            pdf.cell(0, 8, "Графік сценарного прогнозу", ln=1)
            pdf.image(tmp_path, x=15, w=180)
        except Exception:
            pass

    # ---------- ПУАССОН (FIX: точний інтервал і λ з нерозкругленого значення) ----------
    pdf.add_page()
    pdf.set_font(font_name, "B", 13)
    pdf.cell(0, 9, "Прогноз за методом Пуассона", ln=1)

    headers_ci = ["Рік", "Очік. (λ)", "95% ДІ низ", "95% ДІ верх"]
    rows_ci = []
    if pop_used > 0:
        for i, y in enumerate(forecast_years):
            lam = mid_A[i] * pop_used / 100000.0  # float λ, НЕ з округлених abs_mid
            L, U = _poisson_pi_counts(lam, alpha=0.05)
            rows_ci.append([y, int(round(lam)), L, U])
    else:
        rows_ci = [[y, "-", "-", "-"] for y in forecast_years]
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, "Увага: N не задане — ДІ Пуассона за абсолютами неможливо оцінити.")
    _table(pdf, font_name, bold_avail, headers_ci, rows_ci, widths=[epw/4]*4)

    # Графік Пуассона (середній сценарій)
    if poisson_chart_buf:
        try:
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            with open(tmp_path, "wb") as f:
                f.write(poisson_chart_buf.getvalue() if hasattr(poisson_chart_buf, "getvalue") else poisson_chart_buf)
            pdf.set_font(font_name, "B", 12)
            pdf.cell(0, 8, "Графік Пуассона (середній сценарій)", ln=1)
            pdf.image(tmp_path, x=15, w=180)
        except Exception:
            pass

    # ---------- t-Стьюдента ----------
    pdf.add_page()
    pdf.set_font(font_name, "B", 14)
    pdf.cell(0, 8, "Прогноз за методом Стьюдента", ln=1)
    # ... (твій блок Student без змін) ...

    # ---- Висновки ----
    pdf.add_page()
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 7, "Висновки", ln=1)
    pdf.set_font(font_name, "", 11)
    pdf.multi_cell(0, 6, "Сценарії прогнозу та оцінки невизначеності допомагають планувати ресурси системи охорони здоров’я.")

    pdf_bytes = pdf.output(dest="S").tobytes()
    pdf_filename = f"TB_{region}_{start_year}_report.pdf"
    return pdf_bytes, pdf_filename

# =========================
# Хелпери моделі
# =========================
def _pick_base_incidence(df_input: pd.DataFrame, period: str) -> float:
    df_sorted = df_input.sort_values("Рік")
    if "Період" in df_sorted.columns and period in df_sorted["Період"].values:
        return _safe_float(df_sorted[df_sorted["Період"] == period]["Захворюваність"].iloc[-1])
    return _safe_float(df_sorted["Захворюваність"].iloc[-1])

def _pick_base_incidence_with_cutoff(df_input: pd.DataFrame, period: str, cutoff_year: int) -> float:
    df_sorted = df_input.sort_values("Рік")
    df_cut = df_sorted[df_sorted["Рік"] <= _safe_int(cutoff_year)]
    if df_cut.empty:
        return _pick_base_incidence(df_input, period)
    if "Період" in df_cut.columns and period in df_cut["Період"].values:
        return _safe_float(df_cut[df_cut["Період"] == period]["Захворюваність"].iloc[-1])
    return _safe_float(df_cut["Захворюваність"].iloc[-1])

def _make_years(start_year: int, horizon: int) -> list[int]:
    return list(range(_safe_int(start_year), _safe_int(start_year) + _safe_int(horizon)))

def forecast_universal(base_inc: float, horizon: int, rates: dict[str, float]) -> dict[str, list[float]]:
    out = {"opt": [], "mid": [], "pes": []}
    for k in out.keys():
        r = _safe_float(rates.get(k, 0.0)); cur = base_inc; seq = []
        for _ in range(_safe_int(horizon)):
            cur = cur * (1.0 + r)
            seq.append(cur)
        out[k] = seq
    return out

def forecast_postwar(base_inc: float, horizon: int,
                     k1: dict[str, float], k23: dict[str, float], k45: dict[str, float]) -> dict[str, list[float]]:
    out = {"opt": [], "mid": [], "pes": []}
    horizon = _safe_int(horizon)
    for sc in out.keys():
        seq = []
        if horizon >= 1:
            seq.append(base_inc * (1.0 + _safe_float(k1[sc])))
        cur = seq[-1] if seq else base_inc
        for _ in range(max(0, min(horizon, 3) - 1)):
            cur = cur * (1.0 + _safe_float(k23[sc])); seq.append(cur)
        for _ in range(max(0, horizon - 3)):
            cur = cur * (1.0 + _safe_float(k45[sc])); seq.append(cur)
        out[sc] = seq[:horizon]
    return out

def _pick_population_for_abs(period: str) -> int:
    if period == "післявоєнний":
        return _safe_int(st.session_state.get("population_postwar", 0))
    elif period == "до повномасштабного вторгнення":
        pre = _safe_int(st.session_state.get("population_prewar", 0))
        return pre if pre > 0 else _safe_int(st.session_state.get("population_current", 0))
    else:  # «повномасштабне вторгнення» та ін.
        return _safe_int(st.session_state.get("population_current", 0))
# =========================
# 1. Географія (Область → Район → Громада) з ручними опціями для Донецької, Луганської, АР Крим
# =========================
import streamlit as st

st.markdown("### 1. Географія")

REGIONS = [
    "Вінницька","Волинська","Дніпропетровська","Донецька","Житомирська",
    "Закарпатська","Запорізька","Івано-Франківська","Київська","Кіровоградська",
    "Луганська","Львівська","Миколаївська","Одеська","Полтавська","Рівненська",
    "Сумська","Тернопільська","Харківська","Херсонська","Хмельницька",
    "Черкаська","Чернівецька","Чернігівська","м. Київ","м. Севастополь","АР Крим"
]

region = st.selectbox("Оберіть область України:", REGIONS, key="region_select")

# --- РАЙОНИ (додано «Ввести вручну» для Донецької/Луганської, Крим обробляємо окремо)
rayon_map = {
    "Вінницька": ["Аналізується вся область","Вінницький","Гайсинський","Жмеринський","Могилів-Подільський","Тульчинський","Хмільницький"],
    "Волинська": ["Аналізується вся область","Володимирський","Камінь-Каширський","Ковельський","Луцький"],
    "Дніпропетровська": ["Аналізується вся область","Дніпровський","Кам’янський","Криворізький","Нікопольський","Новомосковський","Павлоградський","Синельниківський"],
    "Донецька": ["Аналізується вся область","Ввести вручну"],
    "Житомирська": ["Аналізується вся область","Бердичівський","Житомирський","Звягельський","Коростенський"],
    "Закарпатська": ["Аналізується вся область","Берегівський","Мукачівський","Рахівський","Тячівський","Ужгородський","Хустський"],
    "Запорізька": ["Аналізується вся область","Бердянський","Василівський","Запорізький","Мелітопольський","Пологівський"],
    "Івано-Франківська": ["Аналізується вся область","Івано-Франківський","Калуський","Косівський","Коломийський","Надвірнянський","Верховинський"],
    "Київська": ["Аналізується вся область","Білоцерківський","Бориспільський","Броварський","Бучанський","Вишгородський","Обухівський","Фастівський"],
    "Кіровоградська": ["Аналізується вся область","Голованівський","Кропивницький","Новоукраїнський","Олександрійський"],
    "Луганська": ["Аналізується вся область","Ввести вручну"],
    "Львівська": ["Аналізується вся область","Дрогобицький","Золочівський","Львівський","Самбірський","Стрийський","Червоноградський","Яворівський"],
    "Миколаївська": ["Аналізується вся область","Баштанський","Вознесенський","Миколаївський","Первомайський"],
    "Одеська": ["Аналізується вся область","Білгород-Дністровський","Болградський","Ізмаїльський","Одеський","Подільський","Роздільнянський","Березівський"],
    "Полтавська": ["Аналізується вся область","Кременчуцький","Лубенський","Миргородський","Полтавський"],
    "Рівненська": ["Аналізується вся область","Вараський","Дубенський","Рівненський","Сарненський"],
    "Сумська": ["Аналізується вся область","Конотопський","Охтирський","Роменський","Сумський","Шосткинський"],
    "Тернопільська": ["Аналізується вся область","Кременецький","Тернопільський","Чортківський"],
    "Харківська": ["Аналізується вся область","Ізюмський","Богодухівський","Красноградський","Куп’янський","Лозівський","Харківський","Чугуївський"],
    "Херсонська": ["Аналізується вся область","Бериславський","Генічеський","Каховський","Скадовський","Херсонський"],
    "Хмельницька": ["Аналізується вся область","Кам’янець-Подільський","Хмельницький","Шепетівський"],
    "Черкаська": ["Аналізується вся область","Звенигородський","Золотоніський","Уманський","Черкаський"],
    "Чернівецька": ["Аналізується вся область","Вижницький","Дністровський","Чернівецький"],
    "Чернігівська": ["Аналізується вся область","Корюківський","Ніжинський","Новгород-Сіверський","Прилуцький","Чернігівський"],
    "м. Київ": [],
    "м. Севастополь": [],
    "АР Крим": []  # Крим нижче — ручні поля (район/громада)
}
# ====== БАЗА ГРОМАД ======
COMMUNITIES_BASE = {
    "Дніпропетровська область": {
        "Дніпровський район": ["Дніпровська міська громада", "Китайгородська сільська громада", "Любимівська сільська громада", "Ляшківська сільська громада", "Миколаївська сільська громада", "Могилівська сільська громада", "Новоолександрівська сільська громада", "Новопокровська селищна громада", "Обухівська селищна громада", "Петриківська селищна громада", "Підгородненська міська громада", "Святовасилівська сільська громада", "Слобожанська селищна громада", "Солонянська селищна громада", "Сурсько-Литовська сільська громада", "Царичанська селищна громада", "Чумаківська сільська громада"],
        "Кам’янський район": ["Божедарівська селищна громада", "Верхівцівська міська громада", "Верхньодніпровська міська громада", "Вишнівська селищна громада", "Вільногірська міська громада", "Жовтоводська міська громада", "Затишнянська сільська громада", "Кам’янська міська громада", "Криничанська селищна громада", "Лихівська селищна громада", "П’ятихатська міська громада", "Саксаганська сільська громада"],
        "Криворізький район": ["Апостолівська міська громада", "Глеюватська сільська громада", "Гречаноподівська сільська громада", "Девладівська сільська громада", "Зеленодольська міська громада", "Карпівська сільська громада", "Криворізька міська громада", "Лозуватська сільська громада", "Лопатинська сільська громада", "Новолатівська сільська громада", "Софіївська селищна громада", "Широківська селищна громада", "Гданцівська сільська громада"],
        "Нікопольський район": ["Марганецька міська громада", "Мирівська сільська громада", "Нікопольська міська громада", "Покровська міська громада", "Першотравенська сільська громада", "Томаківська селищна громада", "Червоногригорівська селищна громада", "Шолохівська сільська громада"],
        "Новомосковський район": ["Губиниська селищна громада", "Личківська сільська громада", "Магдалинівська селищна громада", "Новомосковська міська громада", "Перещепинська міська громада", "Піщанська сільська громада", "Чернеччинська сільська громада"],
        "Павлоградський район": ["Вербківська сільська громада", "Межиріцька сільська громада", "Павлоградська міська громада", "Петропавлівська селищна громада", "Троїцька сільська громада", "Юр’ївська селищна громада"],
        "Синельниківський район": ["Васильківська селищна громада", "Вільнянська сільська громада", "Добропільська сільська громада", "Іларіонівська селищна громада", "Криворізька сільська громада", "Маломихайлівська сільська громада", "Межівська селищна громада", "Новопавлівська сільська громада", "Покровська сільська громада", "Раївська сільська громада", "Роздорська селищна громада", "Синельниківська міська громада", "Славгородська селищна громада", "Українська сільська громада", "Великомихайлівська сільська громада", "Зайцівська сільська громада", "Петропавлівська сільська громада"],
    },
    "Донецька область": {
        "Бахмутський район": ["Бахмутська міська громада", "Соледарська міська громада", "Світлодарська міська громада", "Торецька міська громада", "Часовоярська міська громада", "Сіверська міська громада", "Званівська сільська громада", "Серебрянська сільська громада", "Опитненська сільська громада", "Щербинівська сільська громада"],
        "Волноваський район": ["Волноваська міська громада", "Ольгинська селищна громада", "Великоновосілківська селищна громада", "Комарська сільська громада", "Мирненська сільська громада", "Старомлинівська сільська громада"],
        "Краматорський район": ["Краматорська міська громада", "Дружківська міська громада", "Костянтинівська міська громада", "Лиманська міська громада", "Слов’янська міська громада", "Святогірська міська громада", "Олександрівська селищна громада", "Андріївська сільська громада", "Іллінівська сільська громада", "Миколайпільська сільська громада", "Черкаська селищна громада"],
        "Маріупольський район": ["Нікольська селищна громада", "Сартанська селищна громада", "Мангушська селищна громада", "Старокримська селищна громада"],
        "Покровський район": ["Авдіївська міська громада", "Гірницька міська громада", "Добропільська міська громада", "Мирноградська міська громада", "Покровська міська громада", "Родинська міська громада", "Селидівська міська громада", "Українська міська громада", "Білицька селищна громада", "Гродівська селищна громада", "Криворізька сільська громада", "Новогродівська міська громада", "Шахівська сільська громада"],
        "Горлівський район (підконтрольні території)": ["Світлодарська міська громада", "Миронівська селищна громада", "Новолуганська селищна громада"],
    },
    "Житомирська область": {
        "Бердичівський район": ["Андрушівська міська громада", "Бердичівська міська громада", "Вчорайшенська сільська громада", "Гришковецька селищна громада", "Коднянська сільська громада", "Краснопільська сільська громада", "Ружинська селищна громада", "Семенівська сільська громада", "Червоненська селищна громада"],
        "Житомирський район": ["Баранівська міська громада", "Брониківська сільська громада", "Глибочицька сільська громада", "Городоцька сільська громада", "Довбиська селищна громада", "Іршанська селищна громада", "Житомирська міська громада", "Корнинська селищна громада", "Левківська сільська громада", "Любарська селищна громада", "Миропільська селищна громада", "Новоборівська селищна громада", "Озерянківська сільська громада", "Оліївська сільська громада", "Попільнянська селищна громада", "Радомишльська міська громада", "Романівська селищна громада", "Станишівська сільська громада", "Тетерівська сільська громада", "Хорошівська селищна громада", "Черняхівська селищна громада", "Чуднівська міська громада"],
        "Коростенський район": ["Коростенська міська громада", "Лугинська селищна громада", "Малинська міська громада", "Овруцька міська громада", "Олевська міська громада", "Словечанська сільська громада", "Ушомирська сільська громада"],
        "Новоград-Волинський (Звягельський) район": ["Баранівська міська громада", "Дубрівська сільська громада", "Ємільчинська селищна громада", "Новоград-Волинська (Звягельська) міська громада", "Пулинська селищна громада"],
    },
    "Закарпатська область": {
        "Берегівський район": ["Батівська селищна громада", "Берегівська міська громада", "Вилоцька селищна громада", "Виноградівська міська громада", "Великоберезька сільська громада", "Великобийганська сільська громада", "Косоньська сільська громада", "Пийтерфолвівська сільська громада", "Варівська сільська громада"],
        "Мукачівський район": ["Горондівська сільська громада", "Великолучківська сільська громада", "Верхньокоропецька сільська громада", "Жнятинська сільська громада", "Івановецька сільська громада", "Кольчинська селищна громада", "Мукачівська міська громада", "Неліпинська сільська громада", "Нижньоворітська сільська громада", "Полянська сільська громада", "Свалявська міська громада", "Чинадіївська селищна громада"],
        "Рахівський район": ["Великобичківська селищна громада", "Рахівська міська громада", "Ясінянська селищна громада"],
        "Тячівський район": ["Бедевлянська сільська громада", "Дубівська селищна громада", "Нересницька сільська громада", "Солотвинська селищна громада", "Тересвянська селищна громада", "Тячівська міська громада", "Усть-Чорнянська селищна громада"],
        "Ужгородський район": ["Великоберезнянська селищна громада", "Великодобронська сільська громада", "Дубриницька сільська громада", "Кам’яницька сільська громада", "Оноківська сільська громада", "Перечинська міська громада", "Середнянська селищна громада", "Сюртівська сільська громада", "Тарнівська сільська громада", "Ужгородська міська громада", "Холмківська сільська громада", "Чопська міська громада"],
        "Хустський район": ["Боронявська сільська громада", "Драгівська сільська громада", "Іршавська міська громада", "Колочавська сільська громада", "Керецьківська сільська громада", "Міжгірська селищна громада", "Прислопська сільська громада", "Синевирська сільська громада", "Хустська міська громада"],
    },
    "Запорізька область": {
        "Бердянський район": ["Андрівська сільська громада", "Бердянська міська громада", "Осипенківська сільська громада", "Приморська міська громада"],
        "Василівський район": ["Василівська міська громада", "Дніпрорудненська міська громада", "Енергодарська міська громада", "Кам’янсько-Дніпровська міська громада", "Михайлівська селищна громада", "Пришибська сільська громада", "Степногірська селищна громада"],
        "Запорізький район": ["Балабинська селищна громада", "Вільнянська міська громада", "Запорізька міська громада", "Комишуваська селищна громада", "Михайлівська сільська громада", "Новоолександрівська сільська громада", "Новомиколаївська селищна громада", "Павлівська сільська громада", "Петро-Михайлівська сільська громада", "Степненська сільська громада", "Тернуватська селищна громада", "Широківська сільська громада"],
        "Мелітопольський район": ["Мирненська селищна громада", "Мелітопольська міська громада", "Новенська сільська громада", "Олександрівська сільська громада", "Семенівська сільська громада", "Терпіннівська сільська громада", "Якимівська селищна громада"],
        "Пологівський район": ["Більмацька селищна громада", "Гуляйпільська міська громада", "Оріхівська міська громада", "Пологівська міська громада", "Розівська селищна громада", "Токмацька міська громада"],
    },
    "Івано-Франківська область": {
        "Івано-Франківський район": ["Богородчанська селищна громада", "Букачівська селищна громада", "Єзупільська селищна громада", "Івано-Франківська міська громада", "Лисецька селищна громада", "Олешанська сільська громада", "Рогатинська міська громада", "Солотвинська селищна громада", "Тисменицька міська громада", "Угринівська сільська громада", "Ямницька сільська громада"],
        "Калуський район": ["Войнилівська селищна громада", "Вигодська селищна громада", "Долинська міська громада", "Калуська міська громада", "Новицька сільська громада", "Перегінська селищна громада", "Витвицька сільська громада"],
        "Коломийський район": ["Гвіздецька селищна громада", "Городенківська міська громада", "Коломийська міська громада", "Отинійська селищна громада", "Печеніжинська селищна громада", "Снятинська міська громада", "Заболотівська селищна громада"],
        "Косівський район": ["Косівська міська громада", "Кутська селищна громада", "Яблунівська селищна громада"],
        "Надвірнянський район": ["Битківська селищна громада", "Делятинська селищна громада", "Ланчинська селищна громада", "Надвірнянська міська громада", "Переріслянська сільська громада", "Пасічнянська сільська громада", "Поляницька сільська громада"],
        "Верховинський район": ["Верховинська селищна громада", "Білоберізька сільська громада"],
    },
    "Київська область": {
        "Білоцерківський район": ["Білоцерківська міська громада", "Володарська селищна громада", "Гребінківська селищна громада", "Рокитнянська селищна громада", "Сквирська міська громада", "Ставищенська селищна громада", "Таращанська міська громада", "Тетіївська міська громада", "Узинська міська громада"],
        "Бориспільський район": ["Бориспільська міська громада", "Вороньківська сільська громада", "Золочівська сільська громада", "Переяславська міська громада", "Пристолична сільська громада", "Студениківська сільська громада", "Ташанська сільська громада", "Яготинська міська громада"],
        "Броварський район": ["Баришівська селищна громада", "Броварська міська громада", "Великодимерська селищна громада", "Калитянська селищна громада", "Зазимська сільська громада"],
        "Бучанський район": ["Бабинецька селищна громада", "Бородянська селищна громада", "Бучанська міська громада", "Вишгородська міська громада", "Дмитрівська сільська громада", "Іванківська селищна громада", "Ірпінська міська громада", "Коцюбинська селищна громада", "Немішаївська селищна громада", "Пірнівська сільська громада", "Поліська селищна громада", "Петрівська сільська громада", "Славутицька міська громада"],
        "Обухівський район": ["Богуславська міська громада", "Кагарлицька міська громада", "Миронівська міська громада", "Обухівська міська громада", "Ржищівська міська громада", "Українська міська громада"],
        "Фастівський район": ["Боярська міська громада", "Васильківська міська громада", "Глевахівська селищна громада", "Калинівська селищна громада", "Кожанська селищна громада", "Тетіївська міська громада", "Фастівська міська громада"],
    },
    "Кіровоградська область": {
        "Голованівський район": ["Благовіщенська міська громада", "Вільшанська селищна громада", "Голованівська селищна громада", "Добровеличківська селищна громада", "Побузька селищна громада"],
        "Кропивницький район": ["Аджамська сільська громада", "Великосеверинівська сільська громада", "Дмитрівська сільська громада", "Катеринівська сільська громада", "Кетрисанівська сільська громада", "Компаніївська селищна громада", "Кропивницька міська громада", "Новгородківська селищна громада", "Первозванівська сільська громада", "Суботцівська сільська громада"],
        "Новоукраїнський район": ["Бобринецька міська громада", "Долинська міська громада", "Знам’янська міська громада", "Маловисківська міська громада", "Новомиргородська міська громада", "Новоукраїнська міська громада", "Смолінська селищна громада"],
        "Олександрійський район": ["Олександрійська міська громада", "Онуфріївська селищна громада", "Петрівська селищна громада", "Приютівська селищна громада", "Світловодська міська громада"],
    },
    "Луганська область": {
        "Алчевський район": ["Алчевська міська громада", "Брянківська міська громада", "Кіровська міська громада", "Первомайська міська громада"],
        "Довжанський район": ["Довжанська міська громада", "Ровеньківська міська громада", "Сорокинська міська громада"],
        "Луганський район": ["Луганська міська громада", "Лутугинська міська громада", "Молодогвардійська міська громада"],
        "Рубіжанський район": ["Кремінська міська громада", "Рубіжанська міська громада", "Сєвєродонецька міська громада"],
        "Старобільський район": ["Біловодська селищна громада", "Білокуракинська селищна громада", "Марківська селищна громада", "Міловська селищна громада", "Новоайдарська селищна громада", "Новопсковська селищна громада", "Сватівська міська громада", "Старобільська міська громада", "Троїцька селищна громада"],
        "Щастинський район": ["Гірська міська громада", "Попаснянська міська громада", "Щастинська міська громада"],
    },
    "Львівська область": {
        "Дрогобицький район": ["Бориславська міська громада", "Дрогобицька міська громада", "Меденицька селищна громада", "Східницька селищна громада", "Трускавецька міська громада"],
        "Золочівський район": ["Бродівська міська громада", "Бузька міська громада", "Золочівська міська громада", "Красненська селищна громада", "Підкамінська селищна громада"],
        "Львівський район": ["Бібрська міська громада", "Давидівська сільська громада", "Львівська міська громада", "Мурованська сільська громада", "Оброшинська сільська громада", "Перемишлянська міська громада", "Підберізцівська сільська громада", "Солонківська сільська громада", "Щирецька селищна громада"],
        "Самбірський район": ["Добромильська міська громада", "Новокалинівська міська громада", "Ралівська сільська громада", "Рудківська міська громада", "Самбірська міська громада", "Старосамбірська міська громада", "Турківська міська громада"],
        "Стрийський район": ["Гніздичівська селищна громада", "Жидачівська міська громада", "Моршинська міська громада", "Стрийська міська громада", "Ходорівська міська громада"],
        "Червоноградський район": ["Белзька міська громада", "Великомостівська міська громада", "Добротвірська селищна громада", "Радехівська міська громада", "Сокальська міська громада", "Червоноградська міська громада"],
        "Яворівський район": ["Івано-Франківська селищна громада", "Мостиська міська громада", "Новояворівська міська громада", "Судововишнянська міська громада", "Яворівська міська громада"],
    },
    "Миколаївська область": {
        "Баштанський район": ["Баштанська міська громада", "Березнегуватська селищна громада", "Вільнодолинська сільська громада", "Казанківська селищна громада", "Новобузька міська громада", "Прибузька сільська громада", "Снігурівська міська громада"],
        "Вознесенський район": ["Братська селищна громада", "Вознесенська міська громада", "Доманівська селищна громада", "Єланецька селищна громада", "Прибужанівська сільська громада", "Южноукраїнська міська громада"],
        "Миколаївський район": ["Веснянська сільська громада", "Воскресенська селищна громада", "Коблівська сільська громада", "Мішково-Погорілівська сільська громада", "Миколаївська міська громада", "Новоодеська міська громада", "Очаківська міська громада", "Ольшанська селищна громада", "Первомайська сільська громада"],
        "Первомайський район": ["Арбузинська селищна громада", "Благодатненська сільська громада", "Кривоозерська селищна громада", "Первомайська міська громада", "Врадіївська селищна громада"],
    },
    "Одеська область": {
        "Березівський район": ["Березівська міська громада", "Великобуялицька сільська громада", "Іванівська селищна громада", "Миколаївська селищна громада", "Раухівська селищна громада", "Старомаяківська сільська громада", "Степанівська сільська громада", "Ширяївська селищна громада"],
        "Біляївський район": ["Біляївська міська громада", "Великодолинська селищна громада", "Вигодянська сільська громада", "Дальницька сільська громада", "Маяківська сільська громада", "Таїровська селищна громада", "Теплодарська міська громада", "Усатівська сільська громада"],
        "Болградський район": ["Арцизька міська громада", "Болградська міська громада", "Городненська сільська громада", "Кубейська сільська громада", "Тарутинська селищна громада", "Василівська сільська громада"],
        "Ізмаїльський район": ["Ізмаїльська міська громада", "Кілійська міська громада", "Ренійська міська громада", "Саф’янівська сільська громада", "Суворівська селищна громада"],
        "Одеський район": ["Авангардівська селищна громада", "Дачненська сільська громада", "Доброславська селищна громада", "Красносільська сільська громада", "Овідіопольська селищна громада", "Одеська міська громада", "Фонтанська сільська громада", "Чорноморська міська громада", "Южненська міська громада"],
        "Подільський район": ["Ананьївська міська громада", "Балцька міська громада", "Кодимська міська громада", "Любашівська селищна громада", "Окнянська селищна громада", "Подільська міська громада", "Савранська селищна громада", "Слобідська сільська громада"],
        "Роздільнянський район": ["Великомихайлівська селищна громада", "Захарівська селищна громада", "Лиманська селищна громада", "Роздільнянська міська громада", "Великоплосківська сільська громада"],
    },
    "Полтавська область": {
        "Кременчуцький район": ["Горішньоплавнівська міська громада", "Глобинська міська громада", "Градизька селищна громада", "Кам’янопотоківська сільська громада", "Козельщинська селищна громада", "Кременчуцька міська громада", "Недогарківська сільська громада", "Оболонська сільська громада", "Омельницька сільська громада", "Піщанська сільська громада", "Пришибська сільська громада", "Семенівська селищна громада"],
        "Лубенський район": ["Гребінківська міська громада", "Лубенська міська громада", "Оржицька селищна громада", "Пирятинська міська громада", "Хорольська міська громада", "Чорнухинська селищна громада"],
        "Миргородський район": ["Великобагачанська селищна громада", "Гоголівська селищна громада", "Комишнянська селищна громада", "Лазірківська сільська громада", "Миргородська міська громада", "Ромоданівська селищна громада", "Сергіївська сільська громада", "Шишацька селищна громада"],
        "Полтавський район": ["Диканська селищна громада", "Карлівська міська громада", "Коломацька сільська громада", "Котелевська селищна громада", "Машівська селищна громада", "Новоселівська сільська громада", "Опішнянська селищна громада", "Полтавська міська громада", "Решетилівська міська громада", "Терешківська сільська громада", "Чутівська селищна громада"],
    },
    "Рівненська область": {
        "Вараський район": ["Вараська міська громада", "Володимирецька селищна громада", "Зарічненська селищна громада", "Каноницька сільська громада", "Локницька сільська громада", "Полицька сільська громада", "Рафалівська селищна громада"],
        "Дубенський район": ["Дубенська міська громада", "Демидівська селищна громада", "Млинівська селищна громада", "Радивилівська міська громада", "Смизька селищна громада"],
        "Рівненський район": ["Березнівська міська громада", "Великоомелянська сільська громада", "Городоцька сільська громада", "Гощанська селищна громада", "Клеванська селищна громада", "Корецька міська громада", "Корнинська селищна громада", "Костопільська міська громада", "Малолюбашанська сільська громада", "Острозька міська громада", "Рівненська міська громада", "Шпанівська сільська громада"],
        "Сарненський район": ["Вирівська сільська громада", "Дубровицька міська громада", "Степанська селищна громада", "Сарненська міська громада"],
    },
    "Сумська область": {
        "Конотопський район": ["Буринська міська громада", "Дубов’язівська селищна громада", "Конотопська міська громада", "Новослобідська сільська громада", "Попівська сільська громада", "Путивльська міська громада"],
        "Охтирський район": ["Великописарівська селищна громада", "Кириківська селищна громада", "Краснопільська селищна громада", "Охтирська міська громада", "Тростянецька міська громада", "Чупахівська селищна громада"],
        "Роменський район": ["Андріяшівська сільська громада", "Липоводолинська селищна громада", "Недригайлівська селищна громада", "Роменська міська громада", "Хмелівська сільська громада"],
        "Сумський район": ["Бездрицька сільська громада", "Хотінська селищна громада", "Краснопільська селищна громада", "Нижньосироватська сільська громада", "Садівська сільська громада", "Сумська міська громада", "Степанівська селищна громада", "Юнаківська сільська громада"],
        "Шосткинський район": ["Есманьська селищна громада", "Свеська селищна громада", "Шалигинська селищна громада", "Шосткинська міська громада", "Ямпільська селищна громада"],
    },
    "Тернопільська область": {
        "Чортківський район": ["Борщівська міська громада", "Білобожницька сільська громада", "Васильковецька сільська громада", "Гусятинська селищна громада", "Заводська селищна громада", "Заліщицька міська громада", "Колиндянська сільська громада", "Копичинецька міська громада", "Мельнице-Подільська селищна громада", "Нагірянська сільська громада", "Скала-Подільська селищна громада", "Товстенська селищна громада", "Хоростківська міська громада", "Чортківська міська громада"],
        "Кременецький район": ["Великодедеркальська сільська громада", "Кременецька міська громада", "Лановецька міська громада", "Почаївська міська громада", "Шумська міська громада"],
        "Тернопільський район": ["Байковецька сільська громада", "Великоберезовицька селищна громада", "Великогаївська сільська громада", "Збаразька міська громада", "Зборівська міська громада", "Козівська селищна громада", "Підгаєцька міська громада", "Підгороднянська сільська громада", "Скалатська міська громада", "Тернопільська міська громада", "Теребовлянська міська громада", "Білецька сільська громада"],
    },
    "Харківська область": {
        "Богодухівський район": ["Богодухівська міська громада", "Валківська міська громада", "Володимирівська сільська громада", "Золочівська селищна громада", "Коломацька селищна громада", "Краснокутська селищна громада", "Олександрівська сільська громада"],
        "Ізюмський район": ["Балаклійська міська громада", "Барвінківська міська громада", "Борівська селищна громада", "Донецька селищна громада", "Ізюмська міська громада", "Куньєвська сільська громада", "Савинська селищна громада"],
        "Красноградський район": ["Зачепилівська селищна громада", "Красноградська міська громада", "Наталинська сільська громада", "Сахновщинська селищна громада", "Старовірівська сільська громада"],
        "Куп’янський район": ["Великобурлуцька селищна громада", "Дворічанська селищна громада", "Ківшарівська сільська громада", "Куп’янська міська громада", "Курилівська сільська громада", "Петропавлівська сільська громада", "Шевченківська селищна громада"],
        "Лозівський район": ["Біляївська сільська громада", "Близнюківська селищна громада", "Лозівська міська громада", "Первомайська міська громада", "Панютинська селищна громада", "Олексіївська сільська громада"],
        "Харківський район": ["Безлюдівська селищна громада", "Височанська селищна громада", "Дергачівська міська громада", "Люботинська міська громада", "Малоданилівська селищна громада", "Мереф’янська міська громада", "Нововодолазька селищна громада", "Південноміська селищна громада", "Пісочинська селищна громада", "Покотилівська селищна громада", "Роганська селищна громада", "Солоницівська селищна громада", "Харківська міська громада", "Циркунівська сільська громада", "Чкаловська селищна громада"],
        "Чугуївський район": ["Вовчанська міська громада", "Великобурлуцька селищна громада", "Малинівська селищна громада", "Новопокровська селищна громада", "Печенізька селищна громада", "Старосалтівська селищна громада", "Чкаловська селищна громада", "Чугуївська міська громада"],
    },
    "Херсонська область": {
        "Бериславський район": ["Бериславська міська громада", "Борозенська сільська громада", "Високопільська селищна громада", "Калінінська сільська громада", "Милівська сільська громада", "Новорайська сільська громада", "Новоолександрівська сільська громада", "Нововоронцовська селищна громада", "Тягинська сільська громада", "Кочубеївська сільська громада"],
        "Генічеський район": ["Генічеська міська громада", "Іванівська селищна громада", "Новотроїцька селищна громада", "Нижньосірогозька селищна громада", "Асканія-Нова селищна громада"],
        "Каховський район": ["Каховська міська громада", "Григорівська сільська громада", "Зеленопідська сільська громада", "Любимівська селищна громада", "Новокаховська міська громада", "Роздольненська сільська громада", "Таврійська міська громада", "Чаплинська селищна громада"],
        "Скадовський район": ["Бехтерська сільська громада", "Голопристанська міська громада", "Долматівська сільська громада", "Лазурненська селищна громада", "Мирненська селищна громада", "Новомиколаївська сільська громада", "Скадовська міська громада", "Чулаківська сільська громада"],
        "Херсонський район": ["Білозерська селищна громада", "Дар’ївська сільська громада", "Зеленівська сільська громада", "Музиківська сільська громада", "Олешківська міська громада", "Станіславська сільська громада", "Херсонська міська громада", "Ювілейна сільська громада", "Чорнобаївська сільська громада", "Великокопанівська сільська громада"],
    },
    "Хмельницька область": {
        "Кам’янець-Подільський район": ["Антонінська селищна громада", "Баламутівська сільська громада", "Дунаєвецька міська громада", "Жванецька сільська громада", "Китайгородська сільська громада", "Кам’янець-Подільська міська громада", "Маківська сільська громада", "Новоушицька селищна громада", "Слобідсько-Кульчієвецька сільська громада", "Смотричська селищна громада", "Староушицька селищна громада"],
        "Хмельницький район": ["Волочиська міська громада", "Війтовецька селищна громада", "Гвардійська сільська громада", "Городоцька міська громада", "Летичівська селищна громада", "Меджибізька селищна громада", "Чорноострівська селищна громада", "Красилівська міська громада", "Хмельницька міська громада"],
        "Шепетівський район": ["Білогірська селищна громада", "Ізяславська міська громада", "Красилівська селищна громада", "Полонська міська громада", "Славутська міська громада", "Судилківська сільська громада", "Шепетівська міська громада", "Нетішинська міська громада"],
    },
    "Черкаська область": {
        "Звенигородський район": ["Ватутінська міська громада", "Вільшанська сільська громада", "Звенигородська міська громада", "Катеринопільська селищна громада", "Лисянська селищна громада", "Мокрокалигірська сільська громада", "Шполянська міська громада"],
        "Золотоніський район": ["Вознесенська сільська громада", "Драбівська селищна громада", "Золотоніська міська громада", "Зорівська сільська громада", "Новодмитрівська сільська громада", "Чорнобаївська селищна громада"],
        "Уманський район": ["Баштечківська сільська громада", "Бабанська селищна громада", "Буцька селищна громада", "Жашківська міська громада", "Ладижинська сільська громада", "Монастирищенська міська громада", "Уманська міська громада", "Христинівська міська громада"],
        "Черкаський район": ["Білозірська сільська громада", "Будищенська сільська громада", "Канівська міська громада", "Корсунь-Шевченківська міська громада", "Мліївська сільська громада", "Ротмістрівська сільська громада", "Степанківська сільська громада", "Тернівська сільська громада", "Черкаська міська громада", "Чигиринська міська громада"],
        "Черкаський район (продовження)": ["Городищенська міська громада", "Кам’янська міська громада", "Смілянська міська громада"],
    },
    "Чернівецька область": {
        "Вижницький район": ["Берегометська селищна громада", "Вашківецька міська громада", "Вижницька міська громада", "Іспаська сільська громада"],
        "Дністровський район": ["Вашковецька селищна громада", "Вашковецька (Кельменці) селищна громада", "Клішковецька сільська громада", "Кострижівська селищна громада", "Лівинецька сільська громада", "Мамалигівська сільська громада", "Недобоївська сільська громада", "Новоселицька міська громада", "Окнянська сільська громада", "Рукшинська сільська громада", "Сокирянська міська громада", "Ставчанська сільська громада", "Хотинська міська громада"],
        "Чернівецький район": ["Великокучурівська сільська громада", "Глибоцька селищна громада", "Герцаївська міська громада", "Заставнівська міська громада", "Кіцманська міська громада", "Мамаївська сільська громада", "Магальська сільська громада", "Новоселицька селищна громада", "Острицька сільська громада", "Сторожинецька міська громада", "Тереблеченська сільська громада", "Чагорська сільська громада", "Чернівецька міська громада"],
    },
    "Чернігівська область": {
        "Корюківський район": ["Корюківська міська громада", "Менська міська громада", "Сновська міська громада", "Холминська селищна громада"],
        "Новгород-Сіверський район": ["Коропська селищна громада", "Новгород-Сіверська міська громада", "Понорницька селищна громада", "Семенівська міська громада"],
        "Ніжинський район": ["Батуринська міська громада", "Бахмацька міська громада", "Борзнянська міська громада", "Вертіївська сільська громада", "Дмитрівська селищна громада", "Ічнянська міська громада", "Комарівська сільська громада", "Крутівська сільська громада", "Лосинівська селищна громада", "Мринська сільська громада", "Михайло-Коцюбинська сільська громада", "Ніжинська міська громада", "Носівська міська громада", "Талалаївська селищна громада"],
        "Прилуцький район": ["Ічнянська міська громада", "Линовицька селищна громада", "Малодівицька селищна громада", "Прилуцька міська громада", "Срібнянська селищна громада", "Сухополов’янська сільська громада", "Варвинська селищна громада"],
        "Чернігівський район": ["Городнянська міська громада", "Добрянська селищна громада", "Іванівська сільська громада", "Кіптівська сільська громада", "Киїнська сільська громада", "Куликівська селищна громада", "Любецька селищна громада", "Михайло-Коцюбинська селищна громада", "Олишівська селищна громада", "Остерська міська громада", "Ріпкинська селищна громада", "Тупичівська сільська громада", "Чернігівська міська громада"],
    },
}

# ======= Нормалізація назв для ключів/виводу =======
def _norm_region_name(r: str) -> str:
    if not r:
        return r
    if r in ["м. Київ", "м. Севастополь", "АР Крим"]:
        return r
    return r if r.endswith("область") else f"{r} область"

def _norm_district_name(d: str | None) -> str | None:
    if not d or d in ["Аналізується вся область", "Не застосовується для обраної області"]:
        return None
    return d if d.endswith("район") else f"{d} район"

def _build_full_communities(base: dict, rmap: dict) -> dict:
    """Гарантує, що для всіх відомих районів є ключ у COMMUNITIES (навіть як порожній список)."""
    full = {k: dict(v) for k, v in base.items()}
    for r, dlist in rmap.items():
        rkey = _norm_region_name(r)
        if rkey not in full:
            full[rkey] = {}
        for d in dlist:
            if d in ["Аналізується вся область", "Ввести вручну"]:
                continue
            dkey = _norm_district_name(d)
            if dkey and dkey not in full[rkey]:
                full[rkey][dkey] = []
    return full

COMMUNITIES = _build_full_communities(COMMUNITIES_BASE, rayon_map)

# ======= UI-логіка =======
selected_community = None

# Міста-області (район/громада не застосовні)
if region in ["м. Київ", "м. Севастополь"]:
    district = "Не застосовується для обраної області"
    st.selectbox("Район:", [district], index=0, disabled=True, key="district_disabled")
    st.selectbox("Оберіть громаду:", ["Не застосовується"], index=0, disabled=True, key="hromada_disabled")

# АР Крим — повністю ручне введення
elif region == "АР Крим":
    district = st.text_input("Введіть назву району (АР Крим):", key="district_arc").strip()
    selected_community = st.text_input("Введіть назву громади (АР Крим):", key="hromada_arc").strip()

# Звичайні області + спецлогіка для Донецької/Луганської
else:
    # --- Район ---
    district_choice = st.selectbox(
        "Оберіть район:",
        rayon_map.get(region, ["Аналізується вся область"]),
        key="district_select"
    )

    # Обробка вибору району
    if district_choice == "Аналізується вся область":
        district = None  # означає аналіз усієї області
    elif district_choice == "Ввести вручну":
        district = st.text_input("Введіть район вручну:", key="district_manual").strip() or None
    else:
        district = district_choice

    # --- Громада ---
    if district is None:
        # Якщо аналізується вся область — громада неактуальна
        selected_community = "Аналізується вся область"
        st.info("Аналізується вся область.")
    else:
        oblast_key = _norm_region_name(region)
        rayon_key = _norm_district_name(district)
        base_hromadas = COMMUNITIES.get(oblast_key, {}).get(rayon_key, [])

        # Для Донецької/Луганської додамо «Ввести вручну» + завжди «Аналізується весь район»
        hrom_options = ["Аналізується весь район"] + base_hromadas
        if region in ["Донецька", "Луганська"]:
            hrom_options += ["Ввести вручну"]

        hrom_choice = st.selectbox("Оберіть громаду:", hrom_options, key="hromada_select")

        if hrom_choice == "Аналізується весь район":
            selected_community = "Вся громада району"
        elif hrom_choice == "Ввести вручну":
            selected_community = st.text_input("Введіть громаду вручну:", key="hromada_manual").strip()
        else:
            selected_community = hrom_choice

# ======= Збереження у session_state (нормалізовані значення для PDF/імен файлів) =======
oblast_out = _norm_region_name(region)

if region in ["м. Київ", "м. Севастополь"]:
    district_out = "—"
    hromada_out = "—"
elif region == "АР Крим":
    # Уже введені вручну в полях
    district_out = (district or "Без району")
    hromada_out  = (selected_community or "Без громади")
else:
    # Для стандартних областей
    if district is None:
        district_out = "Весь регіон"       # аналіз усієї області
        hromada_out  = "—"
    else:
        district_out = _norm_district_name(district) or "Без району"
        hromada_out  = (selected_community or "Без громади")

st.session_state["region"]  = oblast_out
st.session_state["district"] = district_out
st.session_state["hromada"]  = hromada_out

# Підсумок
with st.expander("Поточний вибір географії", expanded=False):
    st.write(f"**Область:** {st.session_state['region']}")
    st.write(f"**Район:** {st.session_state['district']}")
    st.write(f"**Громада:** {st.session_state['hromada']}")
# =========================
# 2. Захворюваність — авто-період (без «післявоєнний»), selectbox лише після 2025
# =========================
st.markdown("### 4. Захворюваність")
st.caption(
    "Введіть історичні дані (на 100 тис.). Десяткові — через крапку. "
    "Для років <2026 період визначається автоматично, для років ≥2026 — оберіть вручну."
)
# Ініціалізація сховища
if "incidence_data" not in st.session_state:
    st.session_state["incidence_data"] = []

# Безпечні конвертори
def _sint(x, default=None):
    try:
        if x is None: return default
        s = str(x).strip().replace(",", ".")
        if s == "": return default
        return int(float(s))
    except Exception:
        return default
    except Exception:
        pass

def _sfloat(x, default=None):
    try:
        if x is None: return default
        s = str(x).strip().replace(",", ".")
        if s == "": return default
        return float(s)
    except Exception:
        return default
    except Exception:
        pass

# Автовизначення періоду
def _auto_period(y: int):
    y = _sint(y, None)
    if y is None: return None
    if y < 2020: return "до повномасштабного вторгнення"
    if 2020 <= y <= 2021: return "COVID-19"
    if y == 2022: return "початок війни"
    if 2023 <= y <= 2025: return "повномасштабне вторгнення"
    return None  # від 2026 — ручний вибір

# Кольори
_PERIOD_COLORS = {
    "до повномасштабного вторгнення": "#e8f5e9",  # зелений
    "COVID-19": "#ffe6e6",                        # рожевий
    "початок війни": "#fff3cd",                   # жовтий
    "повномасштабне вторгнення": "#f3e8ff",       # фіолетовий
    "інше": "#f5f5f5",
    "": "#f5f5f5",
}

# Легенда
st.markdown(
    """
**Легенда:**
- 🟢 <b>до повномасштабного вторгнення</b> — роки &lt; 2020  
- 🩷 <b>COVID-19</b> — 2020–2021  
- 🟡 <b>початок війни</b> — 2022  
- 🟣 <b>повномасштабне вторгнення</b> — 2023–2025  
- ⚪️ <b>інші роки</b> — обираються вручну (роки ≥ 2026)
    """,
    unsafe_allow_html=True
)
# ---------------- Форма ----------------
with st.form("incidence_form_hybrid", clear_on_submit=True):
    c1, c2, c3 = st.columns([1, 1, 1.4])

    with c1:
        year_input = st.text_input("Рік", key="inc_year_live", placeholder="напр., 2019")
    with c2:
        inc_input = st.text_input("Захворюваність (на 100 тис.)", key="inc_value_live", placeholder="напр., 38.5")

    live_year = _sint(year_input, None)
    live_auto = _auto_period(live_year)

    period_selected = None
    period_options = ["", "до повномасштабного вторгнення", "COVID-19",
                      "початок війни", "повномасштабне вторгнення", "мирний час", "післявоєнний", "інше"]

    with c3:
        if live_year is not None and live_year >= 2026:
            period_selected = st.selectbox("Період (для років ≥2026)", period_options, index=0)
        elif live_year is not None:
            badge_text = live_auto if live_auto else "— період не визначено —"
            badge_color = _PERIOD_COLORS.get(live_auto or "", "#f5f5f5")
            st.markdown(
                f"""<div style="margin-top:6px;padding:6px 10px;border-radius:8px;
                               background:{badge_color};display:inline-block;">
                        <b>Автовизначення:</b> {badge_text}
                    </div>""",
                unsafe_allow_html=True
            )

    add_row = st.form_submit_button("➕ Додати рядок")

    if add_row:
        errs = []
        y = _sint(year_input, None)
        v = _sfloat(inc_input, None)

        if y is None:
            errs.append("Вкажіть коректний рік (ціле число).")
        if v is None or v < 0:
            errs.append("Вкажіть коректну захворюваність (невід’ємне число).")

        if y is not None and y >= 2026:
            per = (period_selected or "").strip()
            if not per:
                errs.append("Оберіть період (для років ≥ 2026 — вручну).")
        else:
            per = live_auto

        if errs:
            for e in errs:
                st.error(e)
        else:
            st.session_state["incidence_data"].append({
                "Рік": int(y),
                "Захворюваність": float(round(v, 1)),
                "Період": per
            })
            st.success("✅ Рядок додано")

# ---------------- Таблиця захворюваності ----------------
import pandas as pd

def _row_style(row):
    color = _PERIOD_COLORS.get(str(row.get("Період", "")).strip(), "")
    return [f"background-color: {color}"] * len(row) if color else [""] * len(row)

if st.session_state["incidence_data"]:
    df_inc = pd.DataFrame(
        st.session_state["incidence_data"],
        columns=["Рік", "Захворюваність", "Період"]
    ).copy()

    df_inc["Рік"] = pd.to_numeric(df_inc["Рік"], errors="coerce").astype("Int64")
    df_inc["Захворюваність"] = pd.to_numeric(df_inc["Захворюваність"], errors="coerce")

    # 🔑 Зберігаємо у session_state для прогнозу
    st.session_state["df_main"] = df_inc

    try:
        st.dataframe(
            df_inc.style.apply(_row_style, axis=1).format({"Захворюваність": "{:.1f}"}),
            hide_index=True, use_container_width=True
        )
    except Exception:
        st.dataframe(df_inc, hide_index=True, use_container_width=True)

    # 🗑 Видалення рядків — охайний варіант
    row_to_delete = st.selectbox("Оберіть рік для видалення:", df_inc["Рік"].astype(str))
    if st.button("🗑 Видалити обраний рядок"):
        idx = df_inc[df_inc["Рік"].astype(str) == row_to_delete].index[0]
        st.session_state["incidence_data"].pop(idx)
        st.rerun()

else:
    st.info("Поки що не додано жодного рядка.")
# =========================
# 3. Категорія випадків
# =========================
st.markdown("### 4. Категорія випадків")
category = st.selectbox(
    "Оберіть категорію",
    ["Нові випадки", "Нові + рецидиви", "Легеневий МБТ+", "Усі форми", "Інше (ввести вручну)"],
)
category_custom = st.text_input("Введіть власну категорію") if category == "Інше (ввести вручну)" else category

# =========================
# 4. Період, що прогнозується
# =========================
st.markdown("### 3. Період, що прогнозується")

# Safety: ensure default forecast_period in session_state
if "forecast_period" not in st.session_state:
    st.session_state["forecast_period"] = "повномасштабне вторгнення"

forecast_period = st.selectbox(
    "Оберіть період для прогнозу:",
    [
        "до повномасштабного вторгнення",
        "повномасштабне вторгнення",
        "післявоєнний",
        "мирний час",
        "інше",
    ],
    index=[
        "до повномасштабного вторгнення",
        "повномасштабне вторгнення",
        "післявоєнний",
        "мирний час",
        "інше",
    ].index(st.session_state.get("forecast_period", "повномасштабне вторгнення"))
)

# Якщо користувач обрав "інше" → показуємо поле вводу
if forecast_period == "інше":
    custom_period = st.text_input("Вкажіть власну назву періоду:", key="custom_period")
    if custom_period.strip():
        forecast_period = custom_period.strip()

# Зберігаємо фінальне значення в session_state
st.session_state["forecast_period"] = forecast_period

# Безпечне створення ключів у session_state
if "_pw_params" not in st.session_state:
    st.session_state["_pw_params"] = {}
if "_univ_rates" not in st.session_state:
    st.session_state["_univ_rates"] = {}
if "_pw_or_w_start_year" not in st.session_state:
    st.session_state["_pw_or_w_start_year"] = None

# Рік початку прогнозу
start_year_str = st.text_input("Рік початку прогнозу", value="", placeholder="введіть рік вручну")
year_ok = False
start_year_num = None
if start_year_str.strip():
    try:
        start_year_num = _safe_int(start_year_str.strip())
        year_ok = True
        st.session_state["_pw_or_w_start_year"] = start_year_num
    except Exception:
        st.error("Рік введено некоректно. Введіть ціле число.")

# === Параметри сценаріїв ===
with st.expander("Підказки та параметри сценаріїв для обраного періоду", expanded=False):
    if forecast_period == "післявоєнний":
        st.info("Потрібні: ≥1 рік «повномасштабного вторгнення» у таблиці + Поточне населення та Очікуване повернення.")
        c1, c2, c3 = st.columns(3)
        with c1:
            k1_opt = st.number_input("Оптимістичний — рік 1, %", value=30.0, step=5.0)
            k23_opt = st.number_input("Оптимістичний — роки 2–3 / рік, %", value=-10.0, step=5.0)
            k45_opt = st.number_input("Оптимістичний — роки 4–5 / рік, %", value=-5.0, step=5.0)
        with c2:
            k1_mid = st.number_input("Середній — рік 1, %", value=60.0, step=5.0)
            k23_mid = st.number_input("Середній — роки 2–3 / рік, %", value=-5.0, step=5.0)
            k45_mid = st.number_input("Середній — роки 4–5 / рік, %", value=-2.0, step=2.0)
        with c3:
            k1_pes = st.number_input("Песимістичний — рік 1, %", value=100.0, step=10.0)
            k23_pes = st.number_input("Песимістичний — роки 2–3 / рік, %", value=5.0, step=5.0)
            k45_pes = st.number_input("Песимістичний — роки 4–5 / рік, %", value=10.0, step=5.0)
        st.session_state["_pw_params"] = {
            "k1":  {"opt": k1_opt/100.0,  "mid": k1_mid/100.0,  "pes": k1_pes/100.0},
            "k23": {"opt": k23_opt/100.0, "mid": k23_mid/100.0, "pes": k23_pes/100.0},
            "k45": {"opt": k45_opt/100.0, "mid": k45_mid/100.0, "pes": k45_pes/100.0},
        }
    else:
        st.info("Для цього режиму достатньо мати ≥1 рік відповідного періоду у таблиці.")
        defaults = {
            "до повномасштабного вторгнення": {"opt": -3.0, "mid": -1.0, "pes": 2.0},
            "повномасштабне вторгнення":     {"opt":  1.0, "mid":  4.0, "pes": 8.0},
            "мирний час":                     {"opt": -2.0, "mid": -1.0, "pes": 2.0},
        }
        d = defaults.get(forecast_period, {"opt": -1.0, "mid": 0.0, "pes": 1.0})
        c1, c2, c3 = st.columns(3)
        with c1:
            u_opt = st.number_input("Оптимістичний / рік, %", value=float(d.get("opt", -1.0)), step=1.0)
        with c2:
            u_mid = st.number_input("Середній / рік, %", value=float(d.get("mid", 0.0)), step=1.0)
        with c3:
            u_pes = st.number_input("Песимістичний / рік, %", value=float(d.get("pes", 1.0)), step=1.0)
        st.session_state["_univ_rates"] = {"opt": u_opt/100.0, "mid": u_mid/100.0, "pes": u_pes/100.0}
# =========================
# 5. Населення (авто-логіка за періодом)
# =========================
st.markdown("### 6. Населення")

def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            if s == "":
                return default
            return int(float(s))
        return default
    except Exception:
        return default

def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
            if x == "":
                return default
            return float(x)
        return default
    except Exception:
        return default


def _set_migration_loss(prewar: int, current: int):
    prewar_i  = _safe_int(prewar, 0)
    current_i = _safe_int(current, 0)
    if prewar_i > 0 and current_i > 0:
        loss = max(0, prewar_i - current_i)
        st.session_state["migration_loss"] = _safe_int(loss)
        st.metric("Міграційні втрати (авто)", f"{loss:,}".replace(",", " "))
        st.caption("Міграція = Довоєнне − Поточне.")
    else:
        st.session_state["migration_loss"] = None

fp = st.session_state.get("forecast_period", "повномасштабне вторгнення")
start_year_used = _safe_int(
    st.session_state.get("_pw_or_w_start_year", st.session_state.get("start_year", 0)),
    0
)
st.session_state["start_year_used"] = start_year_used

for key in ["population_prewar", "population_current", "population_return", "population_postwar"]:
    if key not in st.session_state:
        st.session_state[key] = 0
    else:
        st.session_state[key] = _safe_int(st.session_state[key], 0)

if "population_plain" not in st.session_state:
    st.session_state["population_plain"] = "0.0"
else:
    st.session_state["population_plain"] = f"{_safe_float(st.session_state['population_plain'], 0.0):.1f}"

# --- Авто-логіка ---
if fp not in ["повномасштабне вторгнення", "післявоєнний"]:
    if start_year_used and start_year_used <= 2022:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.population_prewar = _safe_int(
                st.number_input("Довоєнне населення (осіб)", min_value=0, step=1000,
                                value=_safe_int(st.session_state.population_prewar, 0)), 0)
        with c2:
            st.session_state.population_current = _safe_int(
                st.number_input("Поточне населення (осіб)", min_value=0, step=1000,
                                value=_safe_int(st.session_state.population_current, 0)), 0)
        _set_migration_loss(st.session_state.population_prewar, st.session_state.population_current)
        st.session_state["effective_population"] = _safe_int(st.session_state.population_current, 0)
    else:
        st.markdown("**Введіть загальне населення (тис.)**")
        pop_thousands = st.number_input("Загальне населення (тис.)", min_value=0.0, step=0.1,
                                        value=_safe_float(st.session_state.population_plain, 0.0), format="%.1f")
        st.session_state.population_plain = f"{_safe_float(pop_thousands, 0.0):.1f}"
        st.session_state["effective_population"] = _safe_int(_safe_float(pop_thousands, 0.0) * 1000, 0)
        st.session_state["migration_loss"] = None

elif fp == "повномасштабне вторгнення":
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.population_prewar = _safe_int(
            st.number_input("Довоєнне населення (осіб)", min_value=0, step=1000,
                            value=_safe_int(st.session_state.population_prewar, 0)), 0)
    with c2:
        st.session_state.population_current = _safe_int(
            st.number_input("Поточне населення (осіб)", min_value=0, step=1000,
                            value=_safe_int(st.session_state.population_current, 0)), 0)
    _set_migration_loss(st.session_state.population_prewar, st.session_state.population_current)
    st.session_state["effective_population"] = _safe_int(st.session_state.population_current, 0)

elif fp == "післявоєнний":
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.population_prewar = _safe_int(
            st.number_input(
                "Довоєнне населення (осіб)",
                min_value=0,
                step=1000,
                value=_safe_int(st.session_state.population_prewar, 0)
            ),
            0
        )
    with c2:
        st.session_state.population_current = _safe_int(
            st.number_input(
                "Поточне населення (осіб)",
                min_value=0,
                step=1000,
                value=_safe_int(st.session_state.population_current, 0)
            ),
            0
        )
    st.session_state.population_return = _safe_int(
        st.number_input(
            "Очікуване повернення (осіб)",
            min_value=0,
            step=1000,
            value=_safe_int(st.session_state.population_return, 0)
        ),
        0
    )
    postwar = _safe_int(st.session_state.population_current, 0) + _safe_int(st.session_state.population_return, 0)
    st.session_state.population_postwar = postwar
    st.session_state["effective_population"] = _safe_int(postwar, 0)

    st.metric("Післявоєнне населення (авто)", f"{postwar:,}".replace(",", " "))
    st.caption("Післявоєнне = Поточне + Очікуване повернення.")
# =========================
# 6. Зовнішні фактори — розширений список, рекомендовані діапазони, тюнінг
# =========================
import pandas as pd
import streamlit as st

st.markdown("### 8. Зовнішні фактори")
st.caption(
    "Обeріть релевантні фактори. Для кожного наведено типовий вплив та рекомендований діапазон. "
    "У «профі-режимі» відсоток можна налаштувати вручну. «Інше» — фіксовано 5%."
)

# База факторів: дефолт, рекомендований діапазон (мін–макс), коротка примітка
EXTERNAL_FACTORS_META = {
    # Доступ/система
    "Обмежений доступ до медичної допомоги":        {"default": 15, "range": (5, 25),  "note": "Логістика, безпека, руйнування ЛПЗ"},
    "Недостатня кількість медичного персоналу":     {"default": 10, "range": (5, 20),  "note": "Вакансії, вигорання, плинність"},
    "Відсутність профілактичних заходів":           {"default": 10, "range": (5, 20),  "note": "Скринінг, контакт-трекінг, BCG-покриття"},
    "Переривання лікування":                        {"default": 18, "range": (10, 30), "note": "Втрата на етапах ДЛТ/ПТТ"},
    "Низька прихильність до лікування":             {"default": 14, "range": (5, 25),  "note": "Соціальні/поведінкові бар’єри"},

    # Епідеміологія/соціум
    "Високий рівень ВІЛ серед населення":           {"default": 20, "range": (10, 35), "note": "Коінфекція TB/HIV"},
    "Соціально-економічні труднощі":                {"default": 8,  "range": (5, 20),  "note": "Бідність, безробіття, харчування"},
    "Перенаселеність місць проживання":             {"default": 9,  "range": (5, 20),  "note": "Гуртожитки, ПП, укриття"},
    "Низький рівень обізнаності населення":         {"default": 7,  "range": (3, 15),  "note": "Пізнє звернення, стигма"},
    "Сезонні коливання захворюваності":             {"default": 5,  "range": (2, 10),  "note": "Зимово-весняні піки"},

    # Міграція/війна
    "Міграція населення":                           {"default": 12, "range": (5, 25),  "note": "Виїзд/повернення, розрив спостереження"},
    "Військові дії у регіоні":                      {"default": 25, "range": (10, 40), "note": "Безпека, доступ, переміщення"},
    
    # Плейсхолдер для ручного
    "Інше (ввести вручну)":                         {"default": 5,  "range": (5, 5),   "note": "Фіксовано 5%"},
}

# ——— Режим налаштування
pro_mode = st.toggle("Профі-режим: налаштовувати відсотки вручну (в межах діапазонів)", value=False)

# Побудова списку для вибору
options = []
opt_meta = {}
for name, meta in EXTERNAL_FACTORS_META.items():
    d = meta["default"]
    lo, hi = meta["range"]
    label = f"{name} — {d}% (рек.: {lo}–{hi}%)"
    options.append(label)
    opt_meta[label] = {"name": name, "default": d, "range": (lo, hi), "note": meta.get("note", "")}

selected_labels = st.multiselect("Оберіть фактори:", options=options)

# Чи обрано "Інше"
other_selected = any(opt_meta[lbl]["name"].startswith("Інше") for lbl in selected_labels)

# Поле для "Інше" активне лише коли обрано "Інше"
custom_factor_text = st.text_input(
    "Введіть інший фактор (для «Інше», 5% застосовується автоматично):",
    disabled=not other_selected,
)

# Формуємо таблицю
table_rows = []
seen = set()

for lbl in selected_labels:
    meta = opt_meta[lbl]
    base_name = meta["name"]
    dflt = meta["default"]
    lo, hi = meta["range"]
    note = meta["note"]

    if base_name.startswith("Інше"):
        name_final = custom_factor_text.strip() or "Інше (не вказано)"
        key = ("custom", name_final)
        if key not in seen:
            table_rows.append({
                "Фактор": name_final,
                "Вплив (%)": 5,
                "Діапазон (рек.)": "5–5",
                "Примітка": "Фіксовано 5%"
            })
            seen.add(key)
    else:
        key = ("base", base_name)
        if key in seen:
            continue
        if pro_mode:
            perc = st.number_input(
                f"{base_name} — відсоток впливу",
                min_value=float(lo), max_value=float(hi),
                value=float(dflt), step=1.0,
                key=f"extperc::{base_name}"
            )
        else:
            perc = float(dflt)

        table_rows.append({
            "Фактор": base_name,
            "Вплив (%)": int(round(perc)),
            "Діапазон (рек.)": f"{lo}–{hi}",
            "Примітка": note
        })
        seen.add(key)

# Показ таблиці
if table_rows:
    df_ext = pd.DataFrame(table_rows, columns=["Фактор", "Вплив (%)", "Діапазон (рек.)", "Примітка"])
    st.dataframe(df_ext, hide_index=True, use_container_width=True)

    # Сумарний номінальний вплив + попередження, якщо >100%
    total_impact = float(df_ext["Вплив (%)"].sum())
    if total_impact > 100:
        st.warning(f"Сумарний вплив = **{total_impact:.0f}%** (>100%). Перегляньте відсотки або зменште кількість факторів.")
    else:
        st.caption(f"Сумарний номінальний вплив: **{total_impact:.0f}%**")

    # ✅ Збереження для подальших розрахунків
    st.session_state["external_factors_selected"] = df_ext.to_dict(orient="records")
    st.session_state["ext_factor_multiplier"] = 1.0 + total_impact/100.0
else:
    st.info("Не обрано жодного зовнішнього фактора.")
    st.session_state["external_factors_selected"] = []
    st.session_state["ext_factor_multiplier"] = 1.0

# =========================
# 7. Тривалість прогнозу
# =========================
st.markdown("### 7. Тривалість прогнозу")

st.slider(
    "Оберіть кількість років для прогнозу:",
    min_value=1,
    max_value=10,
    value=int(st.session_state.get("duration_years", 5)),
    step=1,
    key="duration_years",
)

# =========================
# 📈 Побудова прогнозу — постійно видима панель (fixed, з build_forecast)
# =========================
import streamlit as st
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math  # для точного Пуассона

# --- Шрифти для графіка (DejaVu Sans, якщо є)
try:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
except Exception:
    pass

# === build_forecast (лог-лінійні сценарії + fallback) ===
def build_forecast(df_main: pd.DataFrame, include_war: bool = False, forecast_horizon: int = 5):
    if df_main is None or df_main.empty:
        return None, None, None

    # копія і санітизація
    df = df_main.copy()
    df["Рік"] = pd.to_numeric(df.get("Рік"), errors="coerce")
    df["Захворюваність"] = pd.to_numeric(df.get("Захворюваність"), errors="coerce")
    df = df.dropna(subset=["Рік", "Захворюваність"])

    # нормалізація «Період» та фільтр воєнних (за потреби)
    if "Період" not in df.columns:
        df["Період"] = ""
    norm_period = df["Період"].astype(str).str.strip().str.lower()
    if not include_war:
        war_periods = {"воєнний", "воєнні", "початок війни", "повномасштабне вторгнення", "війна"}
        df = df[~norm_period.isin(war_periods)]

    if df.empty:
        return None, None, None

    df = df.sort_values("Рік")
    start_year = int(df["Рік"].max()) + 1
    H = max(1, int(forecast_horizon))
    forecast_years = list(range(start_year, start_year + H))

    # лог-лінійний тренд з fallback
    vals = df["Захворюваність"].values.astype(float)
    x = np.arange(len(vals))
    spread = 0.15
    try:
        coeffs = np.polyfit(x, np.log(np.clip(vals, 1e-6, None)), 1)
        slope, intercept = coeffs
        mid = [float(np.exp(intercept + slope * (len(x) + i))) for i in range(H)]
    except Exception:
        if len(vals) >= 2:
            r = float(np.mean(np.diff(np.log(np.clip(vals, 1e-6, None)))))
            mid = [float(vals[-1] * np.exp(r * (i + 1))) for i in range(H)]
        else:
            last = float(vals[-1])
            mid = [last for _ in range(H)]

    opt = [m * (1.0 - spread) for m in mid]
    pes = [m * (1.0 + spread) for m in mid]

    forecast_table = pd.DataFrame({
        "Рік": forecast_years,
        "Оптимістичний": [round(float(x), 1) for x in opt],
        "Середній":       [round(float(x), 1) for x in mid],
        "Песимістичний":  [round(float(x), 1) for x in pes],
    })

    # базовий графік (історія + сценарії)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["Рік"], df["Захворюваність"], "o-", lw=2.2, ms=5, label="Факт")
    ax.plot(forecast_table["Рік"], forecast_table["Оптимістичний"], "--", lw=2.0, label="Оптимістичний")
    ax.plot(forecast_table["Рік"], forecast_table["Середній"], "-", lw=2.4, label="Середній")
    ax.plot(forecast_table["Рік"], forecast_table["Песимістичний"], "--", lw=2.0, label="Песимістичний")
    ax.set_xlabel("Рік")
    ax.set_ylabel("Захворюваність (на 100 тис.)")
    ax.grid(True, alpha=.25)
    ax.legend()

    chart_buf = BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    chart_buf.seek(0)

    return chart_buf, forecast_table, start_year

# === Утиліти ===
def _filter_history_for_include_war(df_hist: pd.DataFrame, include_war: bool) -> pd.DataFrame:
    if df_hist is None or df_hist.empty:
        return df_hist
    df = df_hist.copy()
    if "Період" in df.columns:
        norm = df["Період"].astype(str).str.strip().str.lower()
        war_periods = {"воєнний", "воєнні", "початок війни", "повномасштабне вторгнення", "війна"}
        if not include_war:
            df = df[~norm.isin(war_periods)]
    return df

def _render_chart_from_tables(df_hist: pd.DataFrame, df_forecast: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    if df_hist is not None and not df_hist.empty:
        try:
            ax.plot(df_hist["Рік"], df_hist["Захворюваність"], "o-", color="blue", lw=2.5, ms=6, label="Факт")
        except Exception:
            pass
    if df_forecast is not None and not df_forecast.empty:
        ax.plot(df_forecast["Рік"], df_forecast["Оптимістичний"], "--", color="green", lw=2.0, label="Оптимістичний")
        ax.plot(df_forecast["Рік"], df_forecast["Середній"], "-", color="black", lw=2.8, label="Середній")
        ax.plot(df_forecast["Рік"], df_forecast["Песимістичний"], "--", color="red", lw=2.0, label="Песимістичний")
    ax.set_xlabel("Рік")
    ax.set_ylabel("Захворюваність (на 100 тис.)")
    ax.grid(True, alpha=.3)
    ax.legend()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf

def _render_poisson_chart(df_ci: pd.DataFrame, scenario: str = "Сер") -> BytesIO:
    df_plot = df_ci[df_ci["Сценарій"] == scenario].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.errorbar(
        df_plot["Рік"], df_plot["Очік. випадків"],
        yerr=[df_plot["Очік. випадків"] - df_plot["ДІ низ"],
              df_plot["ДІ верх"] - df_plot["Очік. випадків"]],
        fmt="o-", color="black", ecolor="gray", elinewidth=1.5, capsize=4,
        label=f"{scenario} сценарій"
    )
    ax.set_xlabel("Рік")
    ax.set_ylabel("Абсолютні випадки (з 95% ДІ, Пуассон)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf

def _apply_ext_multiplier(df_forecast: pd.DataFrame) -> pd.DataFrame:
    if df_forecast is None or df_forecast.empty:
        return df_forecast
    m = float(st.session_state.get("ext_factor_multiplier", 1.0) or 1.0)
    m = max(0.5, min(2.0, m))
    out = df_forecast.copy()
    for col in ("Оптимістичний", "Середній", "Песимістичний"):
        if col in out.columns:
            out[col] = (out[col].astype(float) * m).round(1)
    st.session_state["ext_multiplier_used"] = m
    return out

# --- точні квантилі Пуассона (без SciPy) ---
def _poisson_pi_counts(lam: float, alpha: float = 0.05):
    lam = max(float(lam), 0.0)
    if lam == 0.0:
        return 0, 0
    p = math.exp(-lam)
    cdf = p
    lower_q = alpha/2.0
    upper_q = 1.0 - alpha/2.0
    k = 0
    if cdf >= lower_q:
        L = 0
    else:
        while cdf < lower_q:
            k += 1
            p = p * lam / k
            cdf += p
        L = k
    while cdf < upper_q:
        k += 1
        p = p * lam / k
        cdf += p
    U = k
    return int(L), int(U)

def _clean_pop(x):
    s = str(x).replace('\u00a0','').replace(' ', '').replace(',', '')
    try:
        return int(float(s)) if s not in ('', 'None', 'nan') else 0
    except Exception:
        return 0

def _recompute_abs_and_ci():
    ft = st.session_state.get("forecast_table")
    if ft is None or len(ft) == 0:
        return
    pop_raw = st.session_state.get("effective_population", st.session_state.get("population"))
    population = _clean_pop(pop_raw)
    if population <= 0:
        st.warning("⚠️ Населення N не задане/некоректне — абсолютні випадки та ДІ не розраховані.")
        df_abs = ft[["Рік"]].copy()
        df_abs["Опт"] = "-"
        df_abs["Сер"] = "-"
        df_abs["Пес"] = "-"
        st.session_state["forecast_table_abs"] = df_abs
        st.session_state["ci_cases"] = pd.DataFrame(columns=["Рік","Сценарій","Очік. випадків","ДІ низ","ДІ верх"])
        st.session_state["ci_incidence"] = pd.DataFrame(columns=["Рік","Сценарій","Очік. інц.","ДІ низ","ДІ верх"])
        return

    df_inc = ft.copy()
    lam_opt = df_inc["Оптимістичний"].astype(float) * population / 100000.0
    lam_mid = df_inc["Середній"].astype(float)       * population / 100000.0
    lam_pes = df_inc["Песимістичний"].astype(float)  * population / 100000.0

    df_abs = pd.DataFrame({
        "Рік": df_inc["Рік"].astype(int),
        "Опт": lam_opt.round().astype(int),
        "Сер": lam_mid.round().astype(int),
        "Пес": lam_pes.round().astype(int),
    })
    st.session_state["forecast_table_abs"] = df_abs

    rows_cases = []
    for i, y in enumerate(df_abs["Рік"].tolist()):
        for label, lam in [("Опт", float(lam_opt.iloc[i])),
                           ("Сер", float(lam_mid.iloc[i])),
                           ("Пес", float(lam_pes.iloc[i]))]:
            L, U = _poisson_pi_counts(lam, alpha=0.05)
            rows_cases.append({
                "Рік": y,
                "Сценарій": label,
                "Очік. випадків": int(round(lam)),
                "ДІ низ": L,
                "ДІ верх": U
            })
    df_ci_cases = pd.DataFrame(rows_cases)

    def to_inc(x):
        return (float(x) / population) * 100000.0 if population > 0 else 0.0

    rows_inc = []
    for _, r in df_ci_cases.iterrows():
        rows_inc.append({
            "Рік": int(r["Рік"]),
            "Сценарій": r["Сценарій"],
            "Очік. інц.": round(to_inc(r["Очік. випадків"]), 1),
            "ДІ низ": round(to_inc(r["ДІ низ"]), 1),
            "ДІ верх": round(to_inc(r["ДІ верх"]), 1),
        })
    st.session_state["ci_cases"]     = df_ci_cases
    st.session_state["ci_incidence"] = pd.DataFrame(rows_inc)

# === Інтерфейс кнопок ===
st.markdown("### 📈 Побудова прогнозу")

def K(name: str) -> str:
    return f"top:{name}"

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    build_clicked = st.button("📈 Побудувати прогноз", key=K("btn_build"))

with c2:
    if st.button("🔀 Порівняти (воєнні vs без воєнних)", key=K("btn_compare")):
        st.session_state["__run_compare__"] = True

with c3:
    if st.button("🧹 Очистити", key=K("btn_clear")):
        for k in [
            "chart_buf","forecast_table","start_year_used",
            "compare_charts","compare_tables",
            "pdf_bytes","pdf_filename","pdf_obj",
            "forecast_table_abs","ci_cases","ci_incidence",
            "ext_multiplier_used","poisson_chart_buf"
        ]:
            st.session_state.pop(k, None)
        st.success("Очищено.")

# === Обробка кнопки "Побудувати прогноз"
if build_clicked:
    df_main = st.session_state.get("df_main")
    if df_main is None or getattr(df_main, "empty", True):
        st.error("❌ Дані для прогнозу відсутні. Спочатку додайте захворюваність.")
    else:
        if "duration_years" not in st.session_state:
            st.session_state["duration_years"] = int(st.session_state.get("forecast_duration", 5))
        try:
            duration_years = max(1, int(st.session_state.get("duration_years", 5)))
        except Exception:
            duration_years = 5

        try:
            res = build_forecast(
                df_main,
                st.session_state.get("include_war", True),
                duration_years
            )
        except Exception as e:
            res = None
            st.error(f"❌ Помилка побудови прогнозу: {e}")

        if not res or len(res) != 3:
            st.warning("⚠️ Не вдалося побудувати прогноз. Перевірте дані.")
        else:
            chart_buf, forecast_table_raw, start_year_used = res

            df_hist_filtered = _filter_history_for_include_war(
                df_main,
                st.session_state.get("include_war", True)
            )

            forecast_table = _apply_ext_multiplier(forecast_table_raw)

            chart_buf = _render_chart_from_tables(df_hist_filtered, forecast_table)

            from io import BytesIO as _BytesIO
            st.session_state["chart_buf"] = chart_buf.getvalue() if isinstance(chart_buf, _BytesIO) else chart_buf
            st.session_state["forecast_table"] = forecast_table
            st.session_state["start_year_used"] = start_year_used

            _recompute_abs_and_ci()

            try:
                ci_cases = st.session_state.get("ci_cases")
                if ci_cases is not None and not ci_cases.empty:
                    st.session_state["poisson_chart_buf"] = _render_poisson_chart(ci_cases, scenario="Сер")
            except Exception as e:
                st.warning(f"⚠️ Не вдалося згенерувати графік Пуассона: {e}")

            st.success("✅ Прогноз побудовано.")

# === Порівняння прогнозів (war vs nowar)
compare_clicked = st.session_state.pop("__run_compare__", False)

if compare_clicked:
    for k in ["compare_charts", "compare_tables"]:
        st.session_state.pop(k, None)

    df_main = st.session_state.get("df_main")
    if df_main is None or getattr(df_main, "empty", True):
        st.error("❌ Дані для прогнозу відсутні.")
    else:
        duration_years = int(st.session_state.get("duration_years", 5))
        try:
            res_war = build_forecast(df_main, include_war=True,  forecast_horizon=duration_years)
            res_now = build_forecast(df_main, include_war=False, forecast_horizon=duration_years)
        except Exception as e:
            res_war = res_now = None
            st.error(f"❌ Помилка побудови порівняння: {e}")

        if res_war and res_now:
            chart_war, table_war_raw, _ = res_war
            chart_now, table_now_raw, _ = res_now

            df_hist_war   = _filter_history_for_include_war(df_main, include_war=True)
            df_hist_nowar = _filter_history_for_include_war(df_main, include_war=False)

            table_war = _apply_ext_multiplier(table_war_raw)
            table_now = _apply_ext_multiplier(table_now_raw)
            st.session_state["compare_tables"] = (table_war, table_now)

            chart_war = _render_chart_from_tables(df_hist_war, table_war)
            chart_now = _render_chart_from_tables(df_hist_nowar, table_now)

            from io import BytesIO as _BytesIO
            def _to_bytes(b):
                return b.getvalue() if isinstance(b, _BytesIO) else b
            st.session_state["compare_charts"] = (_to_bytes(chart_war), _to_bytes(chart_now))
        else:
            st.warning("⚠️ Не вдалося побудувати обидва прогнози.")

# =========================
# Відображення прогнозу (графік + таблиця) + Метрики (95% ДІ) + Порівняння (war vs nowar)
# =========================
from io import BytesIO
from datetime import datetime
from pathlib import Path
import tempfile, os

st.markdown("### 📊 Прогноз (3 сценарії)")

# --- показ основного графіка прогнозу (якщо вже побудований вище)
_chart_buf = st.session_state.get("chart_buf")
if _chart_buf:
    chart_bytes = _chart_buf.getvalue() if isinstance(_chart_buf, BytesIO) else _chart_buf
    st.image(BytesIO(chart_bytes), use_container_width=True)
    st.download_button(
        "📥 Завантажити PNG",
        data=chart_bytes,
        file_name="forecast_chart.png",
        mime="image/png",
        key="dl_png_main"
    )

    # Таблиця сценаріїв (якщо є)
    forecast_table = st.session_state.get("forecast_table")
    if forecast_table is not None:
        with st.expander("Таблиця сценаріїв", expanded=False):
            st.dataframe(forecast_table, use_container_width=True)

    # ✅ Метрики + 95% ДІ
    if st.session_state.get("ci_incidence") is not None:
        df_ci_i = st.session_state["ci_incidence"]
        df_ci_c = st.session_state.get("ci_cases")

        with st.expander("Метрики та 95% довірчі інтервали (Пуассон)", expanded=False):
            st.caption("Інцидентність (на 100 тис.) — розрахована із ДІ абсолютних випадків (Пуассон).")
            st.dataframe(df_ci_i.sort_values(["Рік","Сценарій"]), use_container_width=True)
            if df_ci_c is not None:
                st.caption("Абсолютні випадки (Пуассон): очікуване значення та 95% ДІ (точні квантилі).")
                st.dataframe(df_ci_c.sort_values(["Рік","Сценарій"]), use_container_width=True)

            # невеликий підсумок по останньому року для середнього сценарію
            try:
                last_year = int(df_ci_i["Рік"].max())
                row_mid = df_ci_i[(df_ci_i["Рік"] == last_year) & (df_ci_i["Сценарій"] == "Сер")].iloc[0]
                st.metric(
                    f"Середній сценарій, інцидентність у {last_year} р.",
                    f"{row_mid['Очік. інц.']:.1f} на 100 тис.",
                    help=f"95% ДІ: {row_mid['ДІ низ']:.1f} — {row_mid['ДІ верх']:.1f}"
                )
            except Exception:
                pass
else:
    st.info("⚠️ Спочатку побудуйте прогноз (секція вище).")

# =========================
# Підготовка даних для прогнозу
# =========================
if "incidence_data" in st.session_state and st.session_state["incidence_data"]:
    df_main = pd.DataFrame(st.session_state["incidence_data"])
else:
    df_main = pd.DataFrame(columns=["Рік", "Захворюваність", "Період"])

# =========================
# Прапорець "Враховувати воєнні роки"
# =========================
include_war = st.checkbox("Враховувати воєнні роки", value=True, key="include_war")

# =========================
# Порівняння прогнозів (war vs nowar) — побудова та відображення, БЕЗ локальної кнопки
# =========================

# тригер із верхньої панелі
compare_clicked = st.session_state.pop("__run_compare__", False)

if compare_clicked:
    # прибрати старі результати
    for k in ["compare_charts", "compare_tables"]:
        st.session_state.pop(k, None)

    df_main = st.session_state.get("df_main")
    if df_main is None or getattr(df_main, "empty", True):
        st.error("❌ Дані для прогнозу відсутні. Завантажте/підготуйте дані вище.")
    else:
        # ТРИВАЛІСТЬ ПРОГНОЗУ: використовуємо єдиний ключ duration_years
        duration_years = int(st.session_state.get("duration_years", 5))
        try:
            # Побудова двох варіантів
            res_war = build_forecast(df_main, include_war=True,  forecast_horizon=duration_years)
            res_now = build_forecast(df_main, include_war=False, forecast_horizon=duration_years)
        except Exception as e:
            res_war = res_now = None
            st.error(f"❌ Помилка побудови порівняння: {e}")

        if res_war and res_now:
            chart_war, table_war_raw, _ = res_war
            chart_now, table_now_raw, _ = res_now

            # 🔗 застосувати зовнішні фактори до обох таблиць
            table_war = _apply_ext_multiplier(table_war_raw)
            table_now = _apply_ext_multiplier(table_now_raw)
            st.session_state["compare_tables"] = (table_war, table_now)

            def _to_bytes(b):
                return b.getvalue() if isinstance(b, BytesIO) else b

            st.session_state["compare_charts"] = (_to_bytes(chart_war), _to_bytes(chart_now))
        else:
            st.warning("⚠️ Не вдалося побудувати обидва прогнози.")

# Показ порівняння, якщо вже є у session_state
if st.session_state.get("compare_charts"):
    st.markdown("### 🔀 Порівняння прогнозів")
    war_b, nowar_b = st.session_state["compare_charts"]

    cL, cR = st.columns(2)
    with cL:
        if war_b:
            st.image(BytesIO(war_b), caption="✅ З урахуванням воєнних років", use_container_width=True)
    with cR:
        if nowar_b:
            st.image(BytesIO(nowar_b), caption="❌ Без урахування воєнних років", use_container_width=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        if war_b:
            st.download_button("📥 PNG (з воєнними)", data=war_b,
                               file_name="compare_war.png", mime="image/png", key="compare:dl_war")
    with dl2:
        if nowar_b:
            st.download_button("📥 PNG (без воєнних)", data=nowar_b,
                               file_name="compare_nowar.png", mime="image/png", key="compare:dl_nowar")
else:
    st.info("⚠️ Спочатку побудуйте прогноз (секція вище), потім натисніть кнопку «Порівняти».")
# =========================
# Формування PDF-звіту (фінальний блок) — без логотипа
# =========================
from datetime import datetime
from io import BytesIO
from pathlib import Path
import tempfile, os
import pandas as pd
import streamlit as st

st.markdown("### ⚙️ Налаштування звіту")

include_chart          = st.session_state.get("opt_include_chart", True)
only_combined          = st.session_state.get("opt_only_combined", False)
include_compare_in_pdf = st.session_state.get("incl_compare_pdf", False)

only_combined = st.checkbox(
    "Стиснути сценарні таблиці (залишити лише комбіновану)",
    value=only_combined, key="opt_only_combined"
)
include_chart = st.checkbox(
    "Додати графік сценарного прогнозування",
    value=include_chart, key="opt_include_chart"
)
include_compare_in_pdf = st.checkbox(
    "Додати порівняння (воєнні vs без воєнних) у PDF",
    value=include_compare_in_pdf, key="incl_compare_pdf"
)

def _clean_pop(x):
    s = str(x).replace('\u00a0','').replace(' ', '').replace(',', '')
    try:
        return int(float(s)) if s not in ('', 'None', 'nan') else 0
    except Exception:
        return 0

if st.session_state.get("forecast_table") is not None:
    duration_years = int(st.session_state.get("duration_years", 5))

    meta = {
        "region":     st.session_state.get("region", "—"),
        "district":   st.session_state.get("district", "—"),
        "hromada":    st.session_state.get("hromada", "—"),
        "period":     st.session_state.get("forecast_period", "—"),
        "start_year": st.session_state.get("start_year_used", "—"),
        "horizon":    duration_years,
    }

    # ---- створення PDF
    pdf = PDFReport(
        "Звіт моделювання туберкульозу",
        meta["region"], meta["district"], meta["hromada"],
        meta["period"], meta["start_year"]
    )

    # ✅ Підключаємо шрифт, який підтримує українську мову
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        pdf.set_font('DejaVu', '', 14)
        pdf._font = 'DejaVu'
    except Exception as e:
        st.warning(f"⚠️ Не вдалося підключити шрифт DejaVuSans: {e}")

    # === титульний блок
    pdf.set_font(pdf._font, "B", 18)
    pdf.cell(0, 10, "Звіт моделювання туберкульозу", ln=1, align="C")
    pdf.set_font(pdf._font, "", 12)
    pdf.cell(0, 8, f"Дата і час формування: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align="C")
    pdf.ln(5)

    pdf.set_font(pdf._font, "", 12)
    pdf.cell(0, 8, f"Область: {meta['region']}", ln=1)
    pdf.cell(0, 8, f"Район: {meta['district']}", ln=1)
    pdf.cell(0, 8, f"Громада: {meta['hromada']}", ln=1)
    pdf.cell(0, 8, f"Період прогнозу: {meta['period']}    Рік початку: {meta['start_year']}", ln=1)
    pdf.cell(0, 8, f"Тривалість прогнозу: {meta['horizon']} років", ln=1)
    pdf._hr(3)

    # ==== зовнішні фактори
    ext_rows = st.session_state.get("external_factors_selected") or []
    ext_mult = float(st.session_state.get("ext_factor_multiplier", 1.0))
    if ext_rows:
        pdf.set_font(pdf._font, "B", 14)
        pdf.cell(0, 8, "Зовнішні фактори впливу", ln=1)
        pdf.set_font(pdf._font, "", 11)

        def safe_text(text: str) -> str:
            text = str(text).replace("\n", " ").replace("\r", " ").strip()
            if not text:
                return "(невідомо)"
            if len(text) > 200:
                text = text[:200] + "..."
            return text

        for row in ext_rows:
            txt = "• "
            if "Фактор" in row and row["Фактор"]:
                txt += safe_text(row["Фактор"])
            else:
                txt += "(невідомо)"
            if "Вплив (%)" in row and str(row["Вплив (%)"]).strip():
                txt += f" — {row['Вплив (%)']}%"

            try:
                pdf.cell(0, 6, safe_text(txt), ln=1)
            except Exception:
                pdf.cell(0, 6, "(помилка відображення рядка)", ln=1)

        pdf.cell(0, 6, f"Застосований множник: ×{ext_mult:.3f}", ln=1)
        pdf._hr(3)

    # ==== сценарні таблиці
    df_inc = st.session_state["forecast_table"].copy()
    if "Рік" in df_inc.columns:
        df_inc = df_inc.sort_values("Рік").reset_index(drop=True)
    df_inc = df_inc.head(duration_years).copy()

    N = _clean_pop(st.session_state.get("effective_population", st.session_state.get("population")))
    df_abs = st.session_state.get("forecast_table_abs")
    if df_abs is None:
        df_abs = df_inc["Рік"].to_frame().copy()
        if N > 0:
            df_abs["Опт"] = (df_inc["Оптимістичний"] * N / 100000).round().astype(int)
            df_abs["Сер"] = (df_inc["Середній"]       * N / 100000).round().astype(int)
            df_abs["Пес"] = (df_inc["Песимістичний"]  * N / 100000).round().astype(int)
        else:
            df_abs["Опт"] = df_abs["Сер"] = df_abs["Пес"] = "-"
        st.session_state["forecast_table_abs"] = df_abs
    else:
        if "Рік" in df_abs.columns:
            df_abs = df_abs.sort_values("Рік").reset_index(drop=True)
        df_abs = df_abs.head(duration_years).copy()

    if only_combined:
        pdf.set_font(pdf._font, "B", 14)
        pdf.cell(0, 8, "Сценарне прогнозування (комбінована таблиця)", ln=1)
        inc_combo = (
            df_inc["Оптимістичний"].map(lambda x: f"{float(x):.1f}") + " / " +
            df_inc["Середній"].map(lambda x: f"{float(x):.1f}") + " / " +
            df_inc["Песимістичний"].map(lambda x: f"{float(x):.1f}")
        )
        cases_combo = (
            df_abs["Опт"].map(str) + " / " +
            df_abs["Сер"].map(str) + " / " +
            df_abs["Пес"].map(str)
        )
        df_combo = pd.DataFrame({"Рік": df_inc["Рік"], "Інц.": inc_combo, "Випадки": cases_combo})
        pdf._table(df_combo, ["Рік", "Інц.", "Випадки"], col_widths=[25, 80, 75])
        pdf._hr(3)
    else:
        pdf.add_scenario_table(df_inc, df_abs)

    # ==== графік сценарію
    if include_chart and st.session_state.get("chart_buf"):
        try:
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            b = st.session_state["chart_buf"]
            Path(tmp_path).write_bytes(b.getvalue() if isinstance(b, BytesIO) else b)
            pdf.add_scenario_chart(tmp_path)
            os.remove(tmp_path)
        except Exception as e:
            st.warning(f"⚠️ Не вдалося додати графік сценарного прогнозування: {e}")

    # ==== Пуассон
    pdf.add_poisson_blocks(df_inc, population=N)
    if st.session_state.get("poisson_chart_buf"):
        try:
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            b = st.session_state["poisson_chart_buf"]
            Path(tmp_path).write_bytes(b.getvalue() if isinstance(b, BytesIO) else b)
            pdf.set_font(pdf._font, "B", 14)
            pdf.cell(0, 8, "Графік прогнозу за Пуассоном (середній сценарій)", ln=1)
            pdf.image(tmp_path, x=pdf.l_margin, w=pdf._content_width())
            pdf._hr(3)
            os.remove(tmp_path)
        except Exception as e:
            st.warning(f"⚠️ Не вдалося додати графік Пуассона: {e}")

    # ==== t-Стьюдента
    df_main = st.session_state.get("df_main")
    if getattr(df_main, "empty", True) is False:
        pdf.add_student_block(df_main)

    # ==== Порівняння
    if include_compare_in_pdf and st.session_state.get("compare_charts"):
        try:
            war_bytes, nowar_bytes = st.session_state["compare_charts"]
            def _dump(buf):
                if not buf: return None
                b = buf.getvalue() if isinstance(buf, BytesIO) else buf
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tf.write(b); tf.close()
                return tf.name
            war_path = _dump(war_bytes)
            nowar_path = _dump(nowar_bytes)
            pdf.add_comparison_page(war_png=war_path, nowar_png=nowar_path)
            for p in [war_path, nowar_path]:
                if p and os.path.exists(p): os.remove(p)
        except Exception:
            pass

    # ==== Висновки
    if hasattr(pdf, "add_conclusions"):
        pdf.add_conclusions()

    # ==== Збереження
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin1")
    elif isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)

    st.session_state["pdf_bytes"] = pdf_bytes
    _now = datetime.now().strftime("%Y-%m-%d")
    clean = lambda s: str(s).replace(" ", "_")
    st.session_state["pdf_filename"] = (
        f"TB_{clean(meta['region'])}_{clean(meta['district'])}_{clean(meta['hromada'])}_"
        f"{_now}_{clean(meta['period'])}_{meta['start_year']}_report.pdf"
    )

else:
    st.info("⚠️ Спочатку побудуйте прогноз, тоді стане доступна генерація PDF.")

# === Кнопка завантаження PDF
if st.session_state.get("pdf_bytes"):
    st.download_button(
        "📥 Завантажити PDF",
        data=st.session_state["pdf_bytes"],
        file_name=st.session_state.get("pdf_filename", "TB_report.pdf"),
        mime="application/pdf",
        key="download_pdf_final",
        use_container_width=True
    )

# 📎 Об'єднання PDF (стабільний варіант із PdfMerger)
# ========================
import streamlit as st
from PyPDF2 import PdfMerger
from io import BytesIO

st.header("📎 Об'єднання PDF")

uploaded_files = st.file_uploader(
    "Завантажте PDF-звіти для об'єднання",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    merger = PdfMerger()
    for file in uploaded_files:
        try:
            merger.append(file)  # приймає file-like одразу
        except Exception as e:
            st.error(f"Не вдалося обробити {file.name}: {e}")

    output_buf = BytesIO()
    try:
        merger.write(output_buf)
    finally:
        merger.close()

    st.download_button(
        "📥 Завантажити об'єднаний PDF",
        data=output_buf.getvalue(),
        file_name="merged_reports.pdf",
        mime="application/pdf"
    )

# =========================
# 📁 Технічний файл (збереження + завантаження)
# =========================
import io, json

st.markdown("## 📁 Робота з технічним файлом")

# --- Збереження ---
with st.expander("💾 Збереження технічного файлу", expanded=False):
    st.caption("Файл міститиме ВСІ введені вами колонки. Підійде і без побудови прогнозу.")
    df_src = st.session_state.get("df_main")
    if isinstance(df_src, pd.DataFrame) and not df_src.empty:
        try:
            df = df_src.copy()
            tech_file = io.BytesIO()
            with pd.ExcelWriter(tech_file, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Data", index=False)
                if st.session_state.get("forecast_table") is not None:
                    st.session_state["forecast_table"].to_excel(writer, sheet_name="Forecast", index=False)
            tech_file.seek(0)

            region_name = st.session_state.get("selected_region") or st.session_state.get("region") or "region"
            file_name = f"tb_forecast_{region_name}.xlsx"
            st.download_button(
                "💾 Завантажити технічний файл",
                data=tech_file,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"❌ Помилка при збереженні технічного файлу: {e}")
    else:
        st.info("ℹ️ Спочатку заповніть дані, щоб зберегти технічний файл.")

# --- Завантаження ---
with st.expander("📂 Завантаження технічного файлу", expanded=False):
    st.caption("Завантажте .xlsx або .json — дані одразу стануть доступні для редагування.")
    uploaded_file = st.file_uploader("Оберіть технічний файл", type=["xlsx", "json"], key="tech_upload")

    if uploaded_file is not None:
        try:
            df_loaded = None
            forecast_loaded = None

            # Якщо JSON
            if uploaded_file.name.lower().endswith(".json"):
                payload = json.load(uploaded_file)
                if isinstance(payload, dict) and "Data" in payload:
                    df_loaded = pd.DataFrame(payload["Data"])
                    if "Forecast" in payload and payload["Forecast"]:
                        forecast_loaded = pd.DataFrame(payload["Forecast"])
                else:
                    df_loaded = pd.DataFrame(payload)
            else:
                # Якщо XLSX
                xls = pd.ExcelFile(uploaded_file)
                if "Data" in xls.sheet_names:
                    df_loaded = pd.read_excel(uploaded_file, sheet_name="Data")
                else:
                    df_loaded = pd.read_excel(uploaded_file, sheet_name=0)

                if "Forecast" in xls.sheet_names:
                    tmp = pd.read_excel(uploaded_file, sheet_name="Forecast")
                    if not tmp.empty:
                        forecast_loaded = tmp

            # --- нормалізація назв колонок ---
            rename_map = {"Year": "Рік", "Incidence": "Захворюваність", "Period": "Період"}
            df_loaded = df_loaded.rename(columns=rename_map)

            for col in ["Рік", "Захворюваність", "Період"]:
                if col not in df_loaded.columns:
                    df_loaded[col] = None

            df_loaded["Рік"] = pd.to_numeric(df_loaded["Рік"], errors="coerce").astype("Int64")
            df_loaded["Захворюваність"] = pd.to_numeric(df_loaded["Захворюваність"], errors="coerce")
            per = df_loaded["Період"]
            per = per.where(per.notna(), "")
            df_loaded["Період"] = per.astype(str).str.strip()

            df_loaded = df_loaded.dropna(subset=["Рік", "Захворюваність"])

            # --- показуємо таблицю з можливістю редагування ---
            edited_df = st.data_editor(
                df_loaded,
                use_container_width=True,
                num_rows="dynamic",
                key="df_main_editor_after_upload",
            )

            # Запис у session_state
            st.session_state["df_main"] = edited_df.copy()
            st.session_state["incidence_data"] = edited_df.to_dict(orient="records")

            if forecast_loaded is not None:
                st.session_state["forecast_table"] = forecast_loaded
                st.info("ℹ️ Додано прогнозні дані з файлу (лист/ключ 'Forecast').")

            st.success("✅ Дані завантажено і доступні для редагування.")

        except Exception as e:
            st.error(f"❌ Помилка при завантаженні технічного файлу: {e}")
