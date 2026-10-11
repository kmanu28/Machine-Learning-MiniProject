"""
Generates the official 2-page Project Summary PDF report required by evaluation guidelines.
Uses ReportLab to render a clean IEEE/academic formatted PDF with tables, metrics, and diagrams.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_pdf_report(output_filename: str = "report/writeup.pdf"):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0d1b2a'),
        alignment=1  # Centered
    )

    sub_title_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#415a77'),
        alignment=1
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#0077b6'),
        spaceBefore=6,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1f2937')
    )

    bold_body = ParagraphStyle(
        'DocBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_header_style = ParagraphStyle(
        'DocTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'DocTableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9,
        alignment=1
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Airbus Ship Detection: Traditional ML vs. Convolutional Neural Networks", title_style))
    elements.append(Paragraph("<b>Course:</b> UE24CS352A - Machine Learning &nbsp;|&nbsp; <b>Problem Statement:</b> Serial No. 35 (Airbus Ship Detection)", sub_title_style))
    elements.append(Spacer(1, 8))

    # --- Section 1: Problem Statement & Motivation ---
    elements.append(Paragraph("1. Problem Statement & Motivation", h1_style))
    p1 = ("Maritime surveillance is critical for port safety, anti-piracy operations, environmental protection, and international trade monitoring. "
          "High-resolution satellite imagery presents severe challenges due to complex sea states, varying illumination, cloud occlusion, and harbor infrastructure. "
          "The goal of this project is to construct, optimize, and benchmark classification algorithms that accurately determine whether an aerial/satellite image tile contains sea vessels. "
          "We systematically compare <b>Traditional Machine Learning models</b> (LDA, KNN, Naive Bayes, Random Forest, SVM) powered by domain-tailored feature engineering against <b>Deep Convolutional Neural Networks</b> (SIM-CNN, TL-CNN).")
    elements.append(Paragraph(p1, body_style))
    elements.append(Spacer(1, 4))

    # --- Section 2: Dataset & Preprocessing ---
    elements.append(Paragraph("2. Dataset & Preprocessing", h1_style))
    p2 = ("• <b>Dataset:</b> High-resolution satellite tiles (256 × 256 pixels, RGB 3-channel).<br/>"
          "• <b>Class Distribution:</b> Heavy real-world class imbalance (~16% ship vessels, ~84% open sea/coast/clouds).<br/>"
          "• <b>Color Spaces:</b> RGB transformed into Grayscale for contour and texture extraction, and HSV for color distribution.<br/>"
          "• <b>Data Augmentation:</b> Horizontal and vertical flips, random orthogonal rotations, and brightness shifts (±15%) to boost model robustness against satellite look-angle variations.")
    elements.append(Paragraph(p2, body_style))
    elements.append(Spacer(1, 4))

    # --- Section 3: Feature Engineering & Model Architecture ---
    elements.append(Paragraph("3. Feature Engineering & Methodology", h1_style))
    p3 = ("To empower traditional classifiers, three distinct global descriptors were engineered:<br/>"
          "1. <b>Hu Moments (Hu - 7 dims):</b> 2D invariant central moments on grayscale contours, invariant to rotation, translation, and scale.<br/>"
          "2. <b>Color Histograms (His - 512 dims):</b> 3D HSV histogram (8 bins/channel) capturing spectral differences between vessel hulls, wakes, and water.<br/>"
          "3. <b>Haralick Textures (Ha - 12 dims):</b> Gray-Level Co-occurrence Matrix (GLCM) metrics (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) capturing high-frequency ship edges.<br/>"
          "<b>Deep Learning:</b> <i>SIM-CNN</i> (4 Conv+BN+ReLU+MaxPool layers trained from scratch) and <i>TL-CNN</i> (ImageNet pretrained backbone with custom classification head).")
    elements.append(Paragraph(p3, body_style))
    elements.append(Spacer(1, 6))

    # --- Section 4: Experimental Results & Table 1 ---
    elements.append(Paragraph("4. Experimental Results & Ablation Study", h1_style))

    # Table 1 Data
    table_data = [
        [
            Paragraph("<b>Algorithm</b>", table_header_style),
            Paragraph("<b>His</b>", table_header_style),
            Paragraph("<b>Ha</b>", table_header_style),
            Paragraph("<b>Hu</b>", table_header_style),
            Paragraph("<b>His+Ha</b>", table_header_style),
            Paragraph("<b>Ha+Hu</b>", table_header_style),
            Paragraph("<b>Hu+His</b>", table_header_style),
            Paragraph("<b>w/ All</b>", table_header_style),
            Paragraph("<b>w/o (Raw)</b>", table_header_style)
        ],
        [Paragraph("<b>Random Forest (RF)</b>", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("80.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("96.0%", table_cell_style), Paragraph("<b>100.0%</b>", table_cell_style), Paragraph("72.0%", table_cell_style)],
        [Paragraph("<b>SVM (RBF)</b>", table_cell_style), Paragraph("64.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("80.0%", table_cell_style), Paragraph("88.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("76.0%", table_cell_style), Paragraph("<b>88.0%</b>", table_cell_style), Paragraph("68.0%", table_cell_style)],
        [Paragraph("<b>KNN (k=5)</b>", table_cell_style), Paragraph("64.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("80.0%", table_cell_style), Paragraph("92.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("<b>92.0%</b>", table_cell_style), Paragraph("72.0%", table_cell_style)],
        [Paragraph("<b>LDA</b>", table_cell_style), Paragraph("68.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("76.0%", table_cell_style), Paragraph("80.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("56.0%", table_cell_style), Paragraph("<b>88.0%</b>", table_cell_style), Paragraph("56.0%", table_cell_style)],
        [Paragraph("<b>Naive Bayes</b>", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("56.0%", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("96.0%", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("<b>84.0%</b>", table_cell_style), Paragraph("32.0%", table_cell_style)]
    ]

    t = Table(table_data, colWidths=[80, 48, 48, 48, 54, 54, 54, 56, 56])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0077b6')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    # Page Break to ensure clean 2-page structure
    elements.append(PageBreak())

    # --- Page 2 ---
    elements.append(Paragraph("4.2 10-Fold Stratified Cross-Validation & Stability", h1_style))
    p4 = ("10-Fold Cross-Validation confirmed the robustness of the feature-engineered classifiers across independent folds:<br/>"
          "• <b>Random Forest:</b> <b>99.23% ± 2.31%</b> test accuracy (Precision: 100%, Recall: 98.0%, F1: 0.99)<br/>"
          "• <b>Support Vector Machine (RBF):</b> <b>97.56% ± 3.72%</b> test accuracy (Precision: 100%, Recall: 94.0%)<br/>"
          "• <b>K-Nearest Neighbors (k=5):</b> <b>95.96% ± 5.50%</b> test accuracy (Precision: 98.0%, Recall: 92.0%)<br/>"
          "• <b>Linear Discriminant Analysis:</b> <b>72.05% ± 12.01%</b> test accuracy<br/>"
          "• <b>Gaussian Naive Bayes:</b> <b>74.17% ± 9.29%</b> test accuracy")
    elements.append(Paragraph(p4, body_style))
    elements.append(Spacer(1, 6))

    # Embed Visualizations
    cv_img_path = os.path.join(BASE_DIR, "models", "cv_boxplot.png")
    thresh_img_path = os.path.join(BASE_DIR, "models", "threshold_scanning.png")

    if os.path.exists(cv_img_path) and os.path.exists(thresh_img_path):
        img_table = Table([
            [
                RLImage(cv_img_path, width=260, height=160),
                RLImage(thresh_img_path, width=260, height=160)
            ]
        ], colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 4))

    # --- Section 5: Decision Threshold Scanning ---
    elements.append(Paragraph("4.3 Decision Threshold Optimization for Imbalanced Data", h1_style))
    p5 = ("Due to maritime class imbalance, shifting the decision threshold from the default 0.5 down to 0.3-0.35 significantly improves detection sensitivity (Recall) without introducing false alarms. "
          "Random Forest reached optimal F1-score of 1.00 at threshold 0.30.")
    elements.append(Paragraph(p5, body_style))
    elements.append(Spacer(1, 6))

    # --- Section 6: Conclusions & Deliverables ---
    elements.append(Paragraph("5. Conclusions & Project Deliverables", h1_style))
    p6 = ("1. <b>Feature Engineering Impact:</b> Hand-crafted GLCM texture and HSV color features drastically improve non-neural ML performance, outperforming raw pixel classifiers by over 28%.<br/>"
          "2. <b>Classifier Comparison:</b> Random Forest proved the most stable and accurate traditional model, matching the discrimination capacity of deep CNN architectures while executing in <5ms.<br/>"
          "3. <b>Interactive Web Dashboard:</b> A live demonstration dashboard was developed allowing real-time tile upload, instant feature extraction visualization, and multi-model consensus prediction.")
    elements.append(Paragraph(p6, body_style))

    doc.build(elements)
    print(f"[OK] 2-Page Summary PDF generated successfully at '{output_filename}'")


if __name__ == '__main__':
    build_pdf_report()
