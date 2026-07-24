"""
Generates a one-page PDF wellness report: current score, ML confidence,
today's inputs, top recommendations, and session-history summary.

Ported from dashboard.py's generate_stress_report(). Behavior is unchanged;
only the auto-install-if-missing bootstrap for reportlab is kept since the
original relied on it rather than requirements.txt.
"""

import io
import subprocess
import sys
import datetime as _dt

import pandas as pd


def _pd_to_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)


def _ensure_reportlab():
    try:
        import reportlab  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "--quiet"])


def generate_stress_report(user, history_df, stress_score, level_name,
                            level_color, sleep, study, screen, anxiety,
                            exercise, pred_proba, LABELS, COLORS) -> bytes:
    _ensure_reportlab()

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=18 * mm, rightMargin=18 * mm,
                             topMargin=14 * mm, bottomMargin=14 * mm)

    LEVEL_HEX = {'Low': '#639922', 'Moderate': '#BA7517', 'High': '#993C1D', 'Critical': '#A32D2D'}
    hex_to_rl = lambda h: rl_colors.HexColor(h)

    lc = hex_to_rl(LEVEL_HEX.get(level_name, '#534AB7'))
    PRIMARY = rl_colors.HexColor('#534AB7')
    ACCENT = rl_colors.HexColor('#D4537E')
    DARK = rl_colors.HexColor('#1E1B3A')
    LIGHT = rl_colors.HexColor('#F4F3FC')
    MUTED = rl_colors.HexColor('#888888')
    WHITE = rl_colors.white
    BLACK = rl_colors.HexColor('#1A1A2E')

    def sty(name, **kw):
        return ParagraphStyle(name, **kw)

    S = {
        'title': sty('title', fontSize=24, textColor=PRIMARY, fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=2),
        'sub': sty('sub', fontSize=10, textColor=MUTED, fontName='Helvetica', alignment=TA_LEFT, spaceAfter=8),
        'h2': sty('h2', fontSize=13, textColor=PRIMARY, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4),
        'body': sty('body', fontSize=9, textColor=BLACK, fontName='Helvetica', spaceAfter=4, leading=14),
        'badge': sty('badge', fontSize=20, textColor=lc, fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=2),
        'score': sty('score', fontSize=11, textColor=MUTED, fontName='Helvetica', spaceAfter=6),
        'centre': sty('centre', fontSize=9, textColor=MUTED, fontName='Helvetica', alignment=TA_CENTER),
        'rec_title': sty('rt', fontSize=9, textColor=BLACK, fontName='Helvetica-Bold', spaceAfter=1),
        'rec_body': sty('rb', fontSize=8, textColor=rl_colors.HexColor('#444444'), fontName='Helvetica', spaceAfter=2, leading=12),
        'footer': sty('footer', fontSize=7, textColor=MUTED, fontName='Helvetica', alignment=TA_CENTER),
    }

    today_str = _dt.date.today().strftime('%A, %d %B %Y')
    username = user.get('username', 'Student').title()

    story = []

    # ── Header band ────────────────────────────────────────────────
    header_data = [[
        Paragraph('<b>Student Stress Monitor</b>', sty('hd', fontSize=14, textColor=WHITE, fontName='Helvetica-Bold')),
        Paragraph(f'Wellness Report · {today_str}', sty('hd2', fontSize=9, textColor=rl_colors.HexColor('#ccccff'), fontName='Helvetica', alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[105 * mm, 65 * mm])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, -1), WHITE),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8 * mm))

    # ── Student + stress result row ────────────────────────────────
    emoji_map = {'Low': 'Low Stress', 'Moderate': 'Moderate Stress',
                 'High': 'High Stress', 'Critical': 'Critical Stress'}
    result_left = [
        [Paragraph('Student', sty('sl', fontSize=8, textColor=MUTED, fontName='Helvetica'))],
        [Paragraph(f'<b>{username}</b>', sty('sn', fontSize=13, textColor=BLACK, fontName='Helvetica-Bold'))],
        [Spacer(1, 4)],
        [Paragraph(f'{emoji_map.get(level_name, level_name)}', S['badge'])],
        [Paragraph(f'Stress Score: <b>{stress_score} / 100</b>', S['score'])],
    ]
    filled = max(1, int(stress_score * 0.82))
    empty = 82 - filled
    bar_data = [[''] * filled + [''] * empty]
    bar_tbl = Table(bar_data, colWidths=[1 * mm] * 82, rowHeights=[4 * mm])
    bar_style = [
        ('BACKGROUND', (0, 0), (filled - 1, 0), lc),
        ('BACKGROUND', (filled, 0), (81, 0), rl_colors.HexColor('#E8E8F0')),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]
    if filled > 0:
        bar_style.append(('ROUNDEDCORNERS', [2]))
    bar_tbl.setStyle(TableStyle(bar_style))
    result_left.append([bar_tbl])

    prob_rows = [[Paragraph('<b>ML Confidence</b>', sty('pc', fontSize=8, textColor=MUTED, fontName='Helvetica-Bold'))]]
    LEVEL_COLS = {'Low': '#639922', 'Moderate': '#BA7517', 'High': '#993C1D', 'Critical': '#A32D2D'}
    if pred_proba is not None:
        for i, lbl in enumerate(LABELS):
            pct = float(pred_proba[i]) * 100
            bar_w = max(1, int(pct * 0.5))
            bc = rl_colors.HexColor(LEVEL_COLS.get(lbl, '#888'))
            mini_bar = Table([[''] * bar_w + [''] * (50 - bar_w)],
                              colWidths=[1 * mm] * 50, rowHeights=[3 * mm])
            mini_bar.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (bar_w - 1, 0), bc),
                ('BACKGROUND', (bar_w, 0), (49, 0), rl_colors.HexColor('#EEEEEE')),
                ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            prob_rows.append([
                Table([[
                    Paragraph(f'{lbl}', sty(f'pl{i}', fontSize=8, textColor=rl_colors.HexColor(LEVEL_COLS.get(lbl, '#888')), fontName='Helvetica-Bold')),
                    mini_bar,
                    Paragraph(f'{pct:.1f}%', sty(f'pp{i}', fontSize=8, textColor=BLACK, fontName='Helvetica')),
                ]], colWidths=[18 * mm, 25 * mm, 12 * mm])
            ])
    else:
        prob_rows.append([Paragraph('Rule-based score used', S['body'])])

    result_tbl = Table(
        [[Table(result_left, colWidths=[88 * mm]), Table(prob_rows, colWidths=[82 * mm])]],
        colWidths=[92 * mm, 82 * mm]
    )
    result_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 1, rl_colors.HexColor('#C5C2E8')),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── Today's inputs grid ────────────────────────────────────────
    story.append(Paragraph("Today's Inputs", S['h2']))
    exer_txt = 'Yes' if int(exercise) >= 1 else 'No'
    input_data = [
        ['Sleep', f'{sleep}h', 'Study', f'{study}h', 'Screen', f'{screen}h'],
        ['Anxiety', f'{anxiety}/10', 'Exercise', exer_txt, 'Score', f'{stress_score}/100'],
    ]
    in_tbl = Table(input_data, colWidths=[22 * mm, 28 * mm, 22 * mm, 28 * mm, 22 * mm, 28 * mm])
    in_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), PRIMARY), ('TEXTCOLOR', (0, 0), (0, -1), WHITE),
        ('BACKGROUND', (2, 0), (2, -1), PRIMARY), ('TEXTCOLOR', (2, 0), (2, -1), WHITE),
        ('BACKGROUND', (4, 0), (4, -1), PRIMARY), ('TEXTCOLOR', (4, 0), (4, -1), WHITE),
        ('BACKGROUND', (1, 0), (1, -1), LIGHT),
        ('BACKGROUND', (3, 0), (3, -1), LIGHT),
        ('BACKGROUND', (5, 0), (5, -1), LIGHT),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#C5C2E8')),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(in_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── Top recommendations ────────────────────────────────────────
    story.append(Paragraph('Top Recommendations', S['h2']))

    RECS = []
    if float(sleep) < 7:
        RECS.append(('High', 'Sleep Below Target',
                      f'Only {sleep}h of sleep. Target 7-9h. Set a fixed bedtime tonight.'))
    if float(study) > 8:
        RECS.append(('High', 'Study Overload',
                      f'{study}h of study today. Sustainable limit is 8h. Schedule a hard stop.'))
    if int(exercise) == 0:
        RECS.append(('Moderate', 'No Exercise Today',
                      'Exercise is the strongest natural stress reducer. Even a 20-min walk helps.'))
    if float(screen) > 4:
        RECS.append(('Moderate', 'High Screen Time',
                      f'{screen}h screen time. Limit to 4h. No screens 30 min before bed.'))
    if int(anxiety) >= 7:
        RECS.append(('Critical', 'High Anxiety',
                      f'Anxiety at {anxiety}/10. Try box breathing: inhale 4s, hold 4s, exhale 4s.'))
    if not RECS:
        RECS.append(('Positive', 'Great Balance',
                      'All key metrics are within healthy ranges. Keep up the consistent habits.'))

    REC_COLORS = {'Critical': '#A32D2D', 'High': '#BA7517', 'Moderate': '#534AB7', 'Positive': '#639922'}
    for sev, title, desc in RECS[:4]:
        rc = rl_colors.HexColor(REC_COLORS.get(sev, '#534AB7'))
        rec_row = Table([[
            Table([[Paragraph(f'<b>{sev}</b>', sty(f'rs{sev}', fontSize=7, textColor=WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER))]], colWidths=[16 * mm], rowHeights=[5 * mm]),
            Table([[Paragraph(f'<b>{title}</b>', S['rec_title'])], [Paragraph(desc, S['rec_body'])]], colWidths=[148 * mm]),
        ]], colWidths=[18 * mm, 150 * mm])
        rec_row.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), rc),
            ('BACKGROUND', (1, 0), (1, 0), rl_colors.HexColor('#F8F7FF')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('BOX', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#C5C2E8')),
        ]))
        story.append(rec_row)
        story.append(Spacer(1, 2 * mm))

    # ── Session history summary ────────────────────────────────────
    if not history_df.empty:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph('Session History Summary', S['h2']))
        hdf = history_df.copy()
        hdf['stress_score'] = _pd_to_numeric(hdf['stress_score'])
        avg_s = hdf['stress_score'].mean()
        sessions = len(hdf)
        best_s = hdf['stress_score'].min()
        worst_s = hdf['stress_score'].max()

        hist_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Total sessions', str(sessions), 'Avg stress score', f'{avg_s:.0f}'],
            ['Best score (lowest)', f'{best_s:.0f}', 'Worst score (highest)', f'{worst_s:.0f}'],
        ]
        hist_tbl = Table(hist_data, colWidths=[42 * mm, 46 * mm, 42 * mm, 42 * mm])
        hist_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('BACKGROUND', (0, 1), (0, -1), LIGHT), ('BACKGROUND', (2, 1), (2, -1), LIGHT),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#C5C2E8')),
        ]))
        story.append(hist_tbl)

    # ── Footer ─────────────────────────────────────────────────────
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=rl_colors.HexColor('#C5C2E8')))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f'Student Stress Monitor · {today_str} · For educational purposes only · Data stored locally',
        S['footer']
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
