# 1. IMPORT
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# 2. KONFIGURASI MODEL
CFG = {
    "IMG_SIZE": (256, 256),   # (height, width)
    "CHANNELS": 3,
    "NUM_CLASSES": 3,
    "CLASSES": ["Hemoragik", "Iskemik", "Normal"],
    "ID_TO_CLASS": {
        0: "Normal / Background",
        1: "Hemoragik",
        2: "Iskemik",
    },
}

CLASS_RGB = {
    0: np.array([46, 204, 113], dtype=np.uint8),   # hijau -> normal / background
    1: np.array([231, 76, 60], dtype=np.uint8),    # merah -> hemoragik
    2: np.array([52, 152, 219], dtype=np.uint8),   # biru  -> iskemik
}

LABEL_STYLE = {
    "Normal": {"color": "#2ecc71", "emoji": "🟢"},
    "Hemoragik": {"color": "#e74c3c", "emoji": "🔴"},
    "Iskemik": {"color": "#3498db", "emoji": "🔵"},
}

MODEL_PATH = "models/attention_unet_stroke_multiclass_nabila.keras"
CONFIDENCE_THRESHOLD_DEFAULT = 0.55

# 3. CUSTOM LAYER, LOSS, & METRIC
@tf.keras.utils.register_keras_serializable(package="StrokeSegmentation")
class ResizeLike(layers.Layer):
    def call(self, inputs):
        source, reference = inputs
        target_size = tf.shape(reference)[1:3]
        return tf.image.resize(source, target_size)


def _dice_score_for_class(y_true, y_pred, class_id, smooth=1e-6):
    y_true_id = tf.argmax(y_true, axis=-1)
    y_pred_id = tf.argmax(y_pred, axis=-1)

    true_class = tf.cast(y_true_id == class_id, tf.float32)
    pred_class = tf.cast(y_pred_id == class_id, tf.float32)

    intersection = tf.reduce_sum(true_class * pred_class)
    total = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class)

    return (2 * intersection + smooth) / (total + smooth)


@tf.keras.utils.register_keras_serializable(package="StrokeSegmentation")
def dice_hemorrhagic(y_true, y_pred):
    return _dice_score_for_class(y_true, y_pred, 1)


@tf.keras.utils.register_keras_serializable(package="StrokeSegmentation")
def dice_ischemic(y_true, y_pred):
    return _dice_score_for_class(y_true, y_pred, 2)


@tf.keras.utils.register_keras_serializable(package="StrokeSegmentation")
def mean_dice(y_true, y_pred):
    return (dice_hemorrhagic(y_true, y_pred) + dice_ischemic(y_true, y_pred)) / 2


@tf.keras.utils.register_keras_serializable(package="StrokeSegmentation")
def mean_iou(y_true, y_pred, smooth=1e-6):
    y_true_id = tf.argmax(y_true, axis=-1)
    y_pred_id = tf.argmax(y_pred, axis=-1)

    scores = []
    for class_id in [1, 2]:
        true_class = tf.cast(y_true_id == class_id, tf.float32)
        pred_class = tf.cast(y_pred_id == class_id, tf.float32)
        intersection = tf.reduce_sum(true_class * pred_class)
        union = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) - intersection
        scores.append((intersection + smooth) / (union + smooth))

    return (scores[0] + scores[1]) / 2


@tf.keras.utils.register_keras_serializable(package="StrokeSegmentation")
def multiclass_ce_dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    class_weight = tf.constant([0.20, 1.00, 1.50], dtype=tf.float32)

    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred) * class_weight, axis=-1)
    ce = tf.reduce_mean(ce)

    dice_losses = []
    for class_id in [1, 2]:
        true_class = y_true[..., class_id]
        pred_class = y_pred[..., class_id]
        intersection = tf.reduce_sum(true_class * pred_class)
        total = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class)
        dice = (2 * intersection + smooth) / (total + smooth)
        dice_losses.append(1 - dice)

    dice_loss = (dice_losses[0] + dice_losses[1]) / 2

    def focal_loss_class(y_true, y_pred, class_id, gamma=2.0, alpha=0.75):
        pred = y_pred[..., class_id]
        true = y_true[..., class_id]
        pt = true * pred + (1 - true) * (1 - pred)
        focal = -alpha * (1 - pt) ** gamma * tf.math.log(pt + 1e-7)
        return tf.reduce_mean(focal)

    focal_ischemic = focal_loss_class(y_true, y_pred, class_id=2, gamma=2.0, alpha=0.75)
    return ce + dice_loss + 0.5 * focal_ischemic


CUSTOM_OBJECTS = {
    "ResizeLike": ResizeLike,
    "multiclass_ce_dice_loss": multiclass_ce_dice_loss,
    "dice_hemorrhagic": dice_hemorrhagic,
    "dice_ischemic": dice_ischemic,
    "mean_dice": mean_dice,
    "mean_iou": mean_iou,
}

# 4. LOAD MODEL
@st.cache_resource(show_spinner="Memuat model Attention U-Net...")
def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

# 5. PREPROCESS & PREDIKSI
def prepare_image(uploaded_file, size=CFG["IMG_SIZE"]):
    height, width = size
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((width, height))

    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_resized, img_array


def class_mask_to_rgb(class_mask):
    rgb = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_RGB.items():
        rgb[class_mask == class_id] = color
    return rgb


def make_overlay(original_rgb, mask_rgb, alpha=0.45):
    original_rgb = np.array(original_rgb)
    overlay = original_rgb.copy().astype(np.float32)
    mask_rgb_f = mask_rgb.astype(np.float32)

    is_lesion = np.any(mask_rgb != CLASS_RGB[0], axis=-1)
    blended = original_rgb.astype(np.float32) * (1 - alpha) + mask_rgb_f * alpha
    overlay[is_lesion] = blended[is_lesion]

    return overlay.astype(np.uint8)


def predict_stroke(model, img_array, confidence_threshold=CONFIDENCE_THRESHOLD_DEFAULT):
    prob = model.predict(img_array, verbose=0)[0]
    pred_mask = np.argmax(prob, axis=-1).astype(np.uint8)
    max_conf = np.max(prob, axis=-1)

    # Piksel lesi dengan confidence rendah dianggap background
    pred_mask[(pred_mask > 0) & (max_conf < confidence_threshold)] = 0

    lesion_pixels = np.sum(pred_mask > 0)

    if lesion_pixels == 0:
        label = "Normal"
        confidence = float(np.mean(max_conf))
    else:
        hemo_pixels = np.sum(pred_mask == 1)
        ischemic_pixels = np.sum(pred_mask == 2)
        label = "Hemoragik" if hemo_pixels >= ischemic_pixels else "Iskemik"
        confidence = float(np.mean(max_conf[pred_mask > 0]))

    return pred_mask, label, confidence


def to_png_bytes(rgb_image):
    bgr = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", bgr)
    return buffer.tobytes() if success else b""

# 6. STREAMLIT CONFIG
st.set_page_config(
    page_title="Stroke CT Segmentation",
    page_icon="🧠",
    layout="wide",
)

# 7. SIDEBAR MENU
st.sidebar.title("Menu")

page = st.sidebar.radio(
    "Pilih Halaman",
    ["Tutorial & Disclaimer", "Prediksi CT Scan"]
)

# 8. HALAMAN 1: TUTORIAL & DISCLAIMER
if page == "Tutorial & Disclaimer":

    st.title("🧠 Sistem Bantu Segmentasi & Klasifikasi Stroke pada CT Scan")
    st.caption("Attention U-Net + EfficientNetB4 — Segmentasi Hemoragik / Iskemik / Normal")

    st.markdown("---")
    st.header("📖 Cara Menggunakan Aplikasi Ini")

    col1, col2 = st.columns(2)

    # Deskripsi Halaman 
    with col1:
        with st.container(border=True):
            st.subheader("📝 Deskripsi Halaman")
            st.markdown(
                """
                Aplikasi ini terdiri dari **2 halaman**, bisa dipilih lewat menu di sidebar kiri:

                1. **Tutorial & Disclaimer** (halaman ini)
                2. **Prediksi CT Scan** — unggah gambar dan lihat hasil analisis
                """
            )

    # Tips 
    with col2:
        with st.container(border=True):
            st.subheader("💡 Tips Agar Hasil Optimal")
            st.markdown(
                """
                - Gunakan gambar CT scan axial (potongan melintang otak).
                - Pastikan gambar cukup jelas / tidak buram.
                - Satu kali unggah = satu slice CT scan.
                """
            )

    col3, col4 = st.columns(2)

    # Langkah-langkah Penggunaan
    with col3:
        with st.container(border=True):
            st.subheader("🚀 Langkah-langkah Penggunaan")
            st.markdown(
                """
                1. Buka halaman **"Prediksi CT Scan"** di sidebar.
                2. Unggah gambar CT scan otak (format **PNG, JPG, atau JPEG**).
                3. Tunggu proses analisis berjalan (beberapa detik).
                4. Sistem menampilkan **3 gambar berdampingan**:
                   - Gambar CT scan asli
                   - Mask hasil segmentasi (area lesi berwarna)
                   - Gambar CT scan yang sudah di-*overlay* dengan mask
                5. Di bawah gambar akan muncul **label klasifikasi akhir**:
                   `Normal`, `Hemoragik`, atau `Iskemik`, beserta *confidence* model.
                6. Anda bisa mengunggah gambar lain kapan saja untuk mengulang proses.
                """
            )

    # Ringkasan Kelas
    with col4:
        with st.container(border=True):
            st.subheader("📊 Ringkasan Kelas yang Dideteksi")
            st.markdown(
                """
                | Warna | Kelas | Keterangan |
                |---|---|---|
                | 🟢 | Normal | Tidak terdeteksi lesi yang meyakinkan |
                | 🔴 | Hemoragik | Diduga terdapat perdarahan |
                | 🔵 | Iskemik | Diduga terdapat sumbatan |
                """
            )

    st.markdown("---")
    st.header("⚠️ Disclaimer Medis — Wajib Dibaca")
    
    st.error(
            """
            **Aplikasi ini BUKAN alat diagnosis medis dan TIDAK menggantikan penilaian tenaga medis profesional.**

            - Hasil segmentasi dan klasifikasi yang ditampilkan berasal dari model *machine learning*
              yang dilatih pada dataset terbatas, dan **berpotensi mengandung kesalahan**
              (false positive maupun false negative).
            - Aplikasi ini ditujukan sebagai **alat bantu penelitian dan skrining awal**, bukan dasar
              tunggal untuk menegakkan diagnosis atau mengubah penanganan medis.
            - **Seluruh keputusan klinis tetap harus diambil oleh dokter/radiolog yang kompeten**,
              dengan mempertimbangkan riwayat klinis pasien dan penilaian langsung terhadap citra asli.
            - Pengembang aplikasi tidak bertanggung jawab atas keputusan medis yang diambil hanya
              berdasarkan keluaran sistem ini.

            Dengan melanjutkan menggunakan aplikasi ini, Anda memahami dan menyetujui hal-hal di atas.
            """
        )

# 9. HALAMAN 2: PREDIKSI CT SCAN
elif page == "Prediksi CT Scan":

    st.title("🧪 Prediksi Segmentasi & Klasifikasi CT Scan")

    st.warning(
        "⚠️ Hasil pada halaman ini adalah keluaran model AI dan **bukan diagnosis medis**. "
        "Selalu konsultasikan dengan dokter / radiolog untuk keputusan klinis."
    )

    # Pengaturan tambahan di sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Pengaturan")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.95,
        value=CONFIDENCE_THRESHOLD_DEFAULT, step=0.05,
        help="Piksel lesi dengan keyakinan di bawah nilai ini dianggap background/normal.",
    )
    overlay_alpha = st.sidebar.slider(
        "Transparansi overlay mask",
        min_value=0.10, max_value=0.90,
        value=0.45, step=0.05,
    )

    # Load model (cached)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(
            f"Gagal memuat model dari '{MODEL_PATH}'. "
            f"Pastikan file model sudah diletakkan di path tersebut. Detail error: {e}"
        )
        st.stop()

    st.subheader("1️⃣ Unggah Gambar CT Scan")

    uploaded_file = st.file_uploader(
        "Pilih file gambar CT scan (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        img_resized, img_array = prepare_image(uploaded_file)

        with st.spinner("Menganalisis gambar..."):
            pred_mask, label, confidence = predict_stroke(
                model, img_array, confidence_threshold=confidence_threshold
            )
            mask_rgb = class_mask_to_rgb(pred_mask)
            overlay_rgb = make_overlay(img_resized, mask_rgb, alpha=overlay_alpha)

        st.markdown("---")
        st.subheader("2️⃣ Hasil Segmentasi")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_resized, caption="Gambar CT Scan Asli", use_container_width=True)

        with col2:
            st.image(mask_rgb, caption="Mask Hasil Segmentasi", use_container_width=True)

        with col3:
            st.image(overlay_rgb, caption="CT Scan + Overlay Mask", use_container_width=True)

        st.caption("🟢 Normal / Background &nbsp;&nbsp; 🔴 Hemoragik &nbsp;&nbsp; 🔵 Iskemik")

        # HASIL LABEL AKHIR
        st.markdown("---")
        st.subheader("3️⃣ Label Klasifikasi Akhir")

        style = LABEL_STYLE.get(label, {"color": "#888", "emoji": "⚪"})

        if label == "Normal":
            st.success(f"Hasil Prediksi: {label}")
        else:
            st.error(f"Hasil Prediksi: {label}")

        st.markdown(
            f"""
            <div style="
                background-color:{style['color']}22;
                border:2px solid {style['color']};
                border-radius:12px;
                padding:16px;
                text-align:center;
                margin-bottom:10px;
            ">
                <span style="font-size:28px;">{style['emoji']}</span>
                <span style="font-size:24px; font-weight:700; color:{style['color']}; margin-left:8px;">
                    {label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(f"Confidence: **{confidence:.4f}**")
        st.progress(float(confidence))

        lesion_pixels = int(np.sum(pred_mask > 0))
        total_pixels = pred_mask.size
        lesion_ratio = (lesion_pixels / total_pixels) * 100
        st.caption(
            f"Luas area terdeteksi sebagai lesi: {lesion_pixels:,} piksel "
            f"({lesion_ratio:.2f}% dari total area gambar)."
        )

        st.error(
            "🔔 **Ingat:** hasil ini adalah alat bantu penelitian/skrining awal, **bukan diagnosis medis**. "
            "Keputusan klinis akhir tetap harus dilakukan oleh dokter atau radiolog yang berkompeten."
        )

        # UNDUH HASIL
        with st.expander("⬇️ Unduh hasil"):
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "Unduh Mask (PNG)",
                    data=to_png_bytes(mask_rgb),
                    file_name="mask_prediksi.png",
                    mime="image/png",
                )
            with dl_col2:
                st.download_button(
                    "Unduh Overlay (PNG)",
                    data=to_png_bytes(overlay_rgb),
                    file_name="overlay_prediksi.png",
                    mime="image/png",
                )
    else:
        st.info("Silakan unggah gambar CT scan untuk memulai analisis.")
