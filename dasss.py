import sklearn
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import plotly_express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import gaussian_kde

# Load dataset
tebet_df = pd.read_excel('dataset/DATA RUMAH TEBET.xlsx')

# Set up Streamlit page configuration
st.set_page_config(page_title='Prediksi Harga Rumah di Tebet', layout='wide')

###################################################################
# 1. Pengantar
st.title('🏠 Laporan Analisis Prediksi Rumah Di Tebet')
st.subheader('1. Pengantar')
st.markdown("""
Analisis ini bertujuan untuk memprediksi harga rumah di daerah Tebet berdasarkan beberapa fitur seperti Luas Bangunan (LB), Luas Tanah (LT), jumlah Kamar Tidur (KT), Kamar Mandi (KM), dan Garasi (GRS). 
Model Polynomial Regression digunakan untuk menangkap hubungan non-linear antara fitur dan target.
""")
st.markdown("---")
###################################################################
# 2. Eksplorasi Data
st.subheader('2. Eksplorasi Data')
st.dataframe(tebet_df)  # Menampilkan DataFrame sebagai tabel

st.markdown("""
**Deskripsi Dataset:**
- Jumlah data: {}
- Fitur: NO, NAMA RUMAH, LB (Luas Bangunan), LT (Luas Tanah), KT (Kamar Tidur), KM (Kamar Mandi), GRS (Garasi)
""".format(tebet_df.shape[0]))

st.markdown("**Deskripsi Data:**")
st.text(tebet_df.describe().to_string())

##############
# Create KDE plot
x = tebet_df['HARGA']
kde = gaussian_kde(x, bw_method=0.5)
x_range = np.linspace(x.min(), x.max(), 100)
kde_values = kde(x_range)

fig_density = go.Figure()
fig_density.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines', name='KDE'))
fig_density.add_trace(go.Histogram(x=x, histnorm='probability density', nbinsx=50, name='Histogram', opacity=0.5))
fig_density.update_layout(title='Plot Density dan Sebaran HARGA', xaxis_title='HARGA', yaxis_title='Density', showlegend=True, bargap=0.1)

st.plotly_chart(fig_density)

##############
def get_features():
    return ['HARGA', 'LB', 'LT', 'KT', 'KM', 'GRS']

# Membuat histogram
def create_histograms(df, features):
    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=[f'Distribusi {feature}' for feature in features])

    # Menambahkan histogram ke subplot
    for i, feature in enumerate(features):
        row = i // 2 + 1  # Menentukan baris
        col = i % 2 + 1   # Menentukan kolom
        histogram = go.Histogram(x=tebet_df[feature], nbinsx=15, name=feature)
        fig.add_trace(histogram, row=row, col=col)

    # Memperbarui layout
    fig.update_layout(height=600, width=1300, title_text="Distribusi Fitur", showlegend=False)
    return fig

# Mendapatkan daftar features
features = get_features()

# Membuat dan menampilkan histogram
fig_histogram = create_histograms(tebet_df, features)
st.plotly_chart(fig_histogram)

# Membuat box plot
def create_box_plots(df, features):
    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=[f'Plot {feature}' for feature in features])

    # Menambahkan box plot ke subplot
    for i, feature in enumerate(features):
        row = i // 2 + 1  # Menentukan baris
        col = i % 2 + 1   # Menentukan kolom
        box_plot = go.Box(y=tebet_df[feature], name=feature)
        fig.add_trace(box_plot, row=row, col=col)

    # Memperbarui layout
    fig.update_layout(height=900, width=1400, title_text="Distribusi Box Plot Fitur", showlegend=False)
    return fig

# Membuat dan menampilkan box plot
fig_boxplot = create_box_plots(tebet_df, features)
st.plotly_chart(fig_boxplot)

##############
# Menghitung korelasi antara setiap fitur dan target 'HARGA'
def correlation_with_target(df, features, target='HARGA'):
    return tebet_df[features].corrwith(tebet_df['HARGA'])

def create_correlation_heatmap(df, features, target='HARGA'):
    # Menghitung korelasi
    correlation_values = correlation_with_target(df, features, target)

    # Mengonversi hasil ke DataFrame untuk keperluan visualisasi
    correlation_df = pd.DataFrame(correlation_values, columns=['Korelasi']).reset_index()
    correlation_df.rename(columns={'index': 'Fitur'}, inplace=True)

    # Membuat heatmap untuk visualisasi korelasi
    fig = go.Figure(data=go.Heatmap(
        z=correlation_df['Korelasi'].values.reshape(1, -1),  # Mengubah ke bentuk 2D untuk heatmap
        x=correlation_df['Fitur'],
        y=['HARGA'],  # Target sebagai label pada sumbu y
        colorscale='Viridis',
        colorbar=dict(title='Nilai Korelasi', ticksuffix='%'),
        zmin=-1, zmax=1
    ))

    # Memperbarui layout
    fig.update_layout(
        title=f'Korelasi Fitur dengan {target}',
        xaxis_title='Fitur',
        yaxis_title='Target',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear')
    )

    return fig

# Membuat dan menampilkan heatmap korelasi
fig_correlation = create_correlation_heatmap(tebet_df, features)
st.plotly_chart(fig_correlation)

def create_correlation_bar(df, features, target='HARGA'):
    # Menghitung korelasi antara fitur-fitur dan target
    correlation_values = correlation_with_target(df, features, target)

    # Mengonversi hasil ke DataFrame untuk keperluan visualisasi
    correlation_df = pd.DataFrame(correlation_values, columns=['Korelasi']).reset_index()
    correlation_df.rename(columns={'index': 'Fitur'}, inplace=True)

    # Membuat bar plot dengan Plotly
    fig = px.bar(correlation_df, x='Fitur', y='Korelasi',
                 labels={'Fitur': 'Fitur', 'Korelasi': 'Korelasi'},
                 title='Korelasi Fitur dengan ' + target)
    fig.update_layout(yaxis_title='Korelasi', xaxis_title='Fitur')
    return fig

# Membuat dan menampilkan bar plot korelasi
fig_correlation_bar = create_correlation_bar(tebet_df, features)
st.plotly_chart(fig_correlation_bar)


##################################################################
# 3. Metodologi
st.markdown("---")
st.subheader('3. Metodologi')
st.markdown("""
**Dataset:**
- Dataset: DATA RUMAH TEBET.xlsx
- Fitur: LB (Luas Bangunan), LT (Luas Tanah), KT (Jumlah Kamar Tidur), KM (Jumlah Kamar Mandi), GRS (Jumlah Garasi) 
- Target: HARGA (Harga Rumah)

**Pembagian Data:**
- Data dibagi menjadi dua bagian: data latih (train) dan data uji (test) dengan rasio 80:20. 
  Ini bertujuan untuk melatih model pada satu subset data dan mengujinya pada subset lainnya untuk mengevaluasi kinerjanya.

**Scaling:**
- Fitur-fitur dinyatakan dalam skala yang sama melalui proses standarisasi menggunakan StandardScaler. 
  Hal ini penting untuk memastikan bahwa model tidak bias terhadap fitur yang memiliki skala lebih besar.

**Model Polynomial Regression:**
- Model Polynomial Regression digunakan untuk menangkap hubungan non-linear antara fitur dan target. 
  Dengan menerapkan PolynomialFeatures, model ini dapat menghasilkan prediksi yang lebih akurat dengan memperhitungkan interaksi antara fitur.
""")

##################################################################
# 4. Hasil Analisis
st.markdown("---")
st.subheader('4. Hasil Analisis')

# Define features and target
X = tebet_df[['LB', 'LT', 'KT', 'KM', 'GRS']]
y = tebet_df['HARGA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
model = Pipeline([
    ('poly', poly),
    ('linear', LinearRegression())
])

# Fit model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.markdown("### Metrik Evaluasi Model")
st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
st.write(f"**R² (Koefisien Determinasi):** {r2:.2f}")

st.markdown("""
**Kesimpulan:**
Model Polynomial Regression menunjukkan performa yang baik dengan R² sebesar {:.2f}, 
yang menunjukkan bahwa model ini dapat menjelaskan {:.2f}% dari variasi harga rumah berdasarkan fitur yang diberikan.
""".format(r2, r2 * 100))

st.markdown("---")

##################################################################
# 5. Input Prediksi
# Sidebar for feature input
st.sidebar.header('Masukkan Fitur untuk Prediksi Harga')
slider_lb = st.sidebar.slider('Luas Bangunan (LB)', 
                               min_value=float(tebet_df['LB'].min()), 
                               max_value=float(tebet_df['LB'].max()), 
                               value=100.0, step=10.0)
slider_lt = st.sidebar.slider('Luas Tanah (LT)', 
                               min_value=float(tebet_df['LT'].min()), 
                               max_value=float(tebet_df['LT'].max()), 
                               value=100.0, step=10.0)
slider_kt = st.sidebar.slider('Jumlah Kamar Tidur (KT)', 
                               min_value=int(tebet_df['KT'].min()), 
                               max_value=int(tebet_df['KT'].max()), 
                               value=2, step=1)
slider_km = st.sidebar.slider('Jumlah Kamar Mandi (KM)',   
                               min_value=int(tebet_df['KM'].min()), 
                               max_value=int(tebet_df['KM'].max()), 
                               value=1, step=1)
slider_grs = st.sidebar.slider('Jumlah Garasi (GRS)', 
                               min_value=int(tebet_df['GRS'].min()), 
                               max_value=int(tebet_df['GRS'].max()), 
                               value=1, step=1)

# Function to predict price
def predict_house_price(lb, lt, kt, km, grs):
    # Create a DataFrame with the input features
    features = pd.DataFrame({
        'LB': [lb],
        'LT': [lt],
        'KT': [kt],
        'KM': [km],
        'GRS': [grs]
    })
    features_scaled = scaler.transform(features)  # Scaling features
    predicted_price = model.predict(features_scaled)
    return predicted_price[0]

# Predict price based on input
if st.sidebar.button('Prediksi Harga'):
    predicted_price = predict_house_price(slider_lb, slider_lt, slider_kt, slider_km, slider_grs)
    st.write(f"Harga rumah impian Anda diperkirakan sekitar IDR {predicted_price:,.2f}")

# Closing remarks
st.markdown("---")
st.markdown("**Terima kasih telah menggunakan aplikasi ini!**")

st.markdown("***Sumber Referensi https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah***")
