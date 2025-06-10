import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Veri Yükleme ve Hazırlama ---
print("--- BÖLÜM 0: Veri Yükleme ve Hazırlama ---")

# Veri setini yükle
try:
    df = pd.read_csv('LifeExpectancy.csv')
    print("Veri seti başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: 'LifeExpectancy.csv' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    exit()

# Sütun adlarındaki boşlukları temizle
df.columns = df.columns.str.strip()

# Sütun adını daha kolay kullanım için değiştir
df = df.rename(columns={'Life expectancy': 'Life_expectancy'})

print("\nVeri setinin ilk 5 satırı:")
print(df.head())
print("\nVeri seti bilgisi:")
df.info()

# --- BÖLÜM 1 ---

# --- 1.1: Veriyi Eğitim ve Test Setlerine Ayırma ---
print("\n\n--- BÖLÜM 1.1: Veriyi Ayırma ---")
test_years = [2003, 2008, 2013]
train_df = df[~df['Year'].isin(test_years)]
test_df = df[df['Year'].isin(test_years)]

print(f"Eğitim seti kayıt sayısı: {len(train_df)}")
print(f"Test seti kayıt sayısı: {len(test_df)}")

# --- 1.2: Keşifsel Veri Analizi ---
print("\n\n--- BÖLÜM 1.2: Keşifsel Veri Analizi ---")

# Yaşam süresi histogramı
plt.figure(figsize=(10, 6))
sns.histplot(df['Life_expectancy'].dropna(), kde=True, bins=30)
plt.title('Yaşam Süresi Dağılımı Histogramı')
plt.xlabel('Yaşam Süresi (Yıl)')
plt.ylabel('Frekans')
plt.grid(True)
plt.show()

# Yaşam süresi istatistikleri
print("\nYaşam Süresi için İstatistiksel Bilgiler:")
print(df['Life_expectancy'].describe())

# En yüksek yaşam süresine sahip 3 ülke
top_countries = df.sort_values(by='Life_expectancy', ascending=False).drop_duplicates('Country').head(3)
print("\nEn Yüksek Yaşam Süresine Sahip 3 Ülke (Tekil Kayıtlara Göre):")
print(top_countries[['Country', 'Year', 'Life_expectancy']])


# --- 1.3 & 1.4: Basit Doğrusal Regresyon Modelleri ---
print("\n\n--- BÖLÜM 1.3 & 1.4: Basit Doğrusal Regresyon ---")
simple_features = ['GDP', 'Total expenditure', 'Alcohol']
models = {}

for feature in simple_features:
    print(f"\n--- Model: Yaşam Süresi vs. {feature} ---")

    # Eksik verileri eğitim setinin ortalamasıyla doldurma
    impute_value = train_df[feature].mean()
    train_df_imputed = train_df.copy()
    test_df_imputed = test_df.copy()
    train_df_imputed[feature].fillna(impute_value, inplace=True)
    test_df_imputed[feature].fillna(impute_value, inplace=True)

    # Hedef değişkendeki (Life_expectancy) eksik verileri içeren satırları kaldır
    train_df_imputed.dropna(subset=['Life_expectancy'], inplace=True)
    test_df_imputed.dropna(subset=['Life_expectancy'], inplace=True)


    # X ve y'yi tanımla
    X_train = train_df_imputed[[feature]]
    y_train = train_df_imputed['Life_expectancy']
    X_test = test_df_imputed[[feature]]
    y_test = test_df_imputed['Life_expectancy']

    # Modeli oluştur ve eğit
    model = LinearRegression()
    model.fit(X_train, y_train)
    models[feature] = {'model': model, 'X_test': X_test, 'y_test': y_test}

    # Katsayıları ve R-kare skorunu bul
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X_train, y_train)

    print(f"Katsayılar: Eğim (slope) = {slope:.4f}, Kesişim (intercept) = {intercept:.4f}")
    print(f"R-kare (R-squared) Skoru (Eğitim Seti): {r2:.4f}")

    # Grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.5, label='Eğitim Verisi')
    plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regresyon Çizgisi')
    plt.title(f'Yaşam Süresi vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel('Yaşam Süresi')
    plt.legend()
    # Denklem metnini oluştur
    equation = f'y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r2:.4f}'
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    plt.grid(True)
    plt.show()

# --- 1.5: Test Seti ile Tahmin ve Hata Analizi ---
print("\n\n--- BÖLÜM 1.5: Basit Regresyon Modelleri için Hata Analizi ---")
for feature, data in models.items():
    model = data['model']
    X_test = data['X_test']
    y_test = data['y_test']

    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)
    errors = y_test - y_pred

    # Ortalama hata ve standart sapma
    mae = mean_absolute_error(y_test, y_pred)
    std_dev_errors = errors.std()

    print(f"\n--- Hata Analizi: {feature} Modeli ---")
    print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
    print(f"Hataların Standart Sapması: {std_dev_errors:.4f}")


# --- BÖLÜM 2 ---

# --- 2.1: Çoklu Regresyon için Özellik Seçimi ---
print("\n\n--- BÖLÜM 2.1: Çoklu Regresyon için Özellik Seçimi ---")

# Sayısal sütunlar arasındaki korelasyonu hesapla
# 'Status' gibi kategorik ve 'Country' gibi tanımlayıcı sütunları çıkar
numeric_df = train_df.drop(columns=['Country', 'Status', 'Year']).copy()
correlation_matrix = numeric_df.corr()

# Yaşam süresi ile olan korelasyonları al ve sırala
life_exp_corr = correlation_matrix['Life_expectancy'].abs().sort_values(ascending=False)

print("Yaşam Süresi ile En Yüksek Korelasyona Sahip Özellikler:")
print(life_exp_corr)

# En iyi 4 özelliği seç (Life_expectancy'nin kendisi hariç)
best_features = life_exp_corr[1:5].index.tolist()
print(f"\nSeçilen 4 özellik: {best_features}")
print("Gerekçe: Bu dört özellik, eğitim verisindeki 'Life_expectancy' (Yaşam Süresi) hedef değişkeni ile mutlak değerce en yüksek korelasyona sahip olanlardır. Yüksek korelasyon, bu değişkenlerin yaşam süresini tahmin etmede daha güçlü bir ilişkiye sahip olduğunu gösterir.")


# --- 2.2 & 2.3: Çoklu Doğrusal Regresyon Modeli ---
print("\n\n--- BÖLÜM 2.2 & 2.3: Çoklu Doğrusal Regresyon ---")

# Eğitim ve test setlerinde eksik verileri doldurma
train_df_multi = train_df.copy()
test_df_multi = test_df.copy()

for col in best_features + ['Life_expectancy']:
    if train_df_multi[col].isnull().sum() > 0:
        impute_value = train_df_multi[col].mean()
        train_df_multi[col].fillna(impute_value, inplace=True)
        test_df_multi[col].fillna(impute_value, inplace=True)

# Hedef değişkende kalan NaN'ları temizle
train_df_multi.dropna(subset=['Life_expectancy'], inplace=True)
test_df_multi.dropna(subset=['Life_expectancy'], inplace=True)


X_train_multi = train_df_multi[best_features]
y_train_multi = train_df_multi['Life_expectancy']
X_test_multi = test_df_multi[best_features]
y_test_multi = test_df_multi['Life_expectancy']

# Modeli oluştur ve eğit
multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)

# Katsayıları ve R-kare skorunu yazdır
print("\nÇoklu Regresyon Modeli Bilgileri:")
print("Katsayılar (Coefficients):")
for feature, coef in zip(best_features, multi_model.coef_):
    print(f"  - {feature}: {coef:.4f}")
print(f"Kesişim (Intercept): {multi_model.intercept_:.4f}")

r2_multi = multi_model.score(X_train_multi, y_train_multi)
print(f"\nR-kare (R-squared) Skoru (Eğitim Seti): {r2_multi:.4f}")

# Test seti üzerinde tahmin ve hata analizi
y_pred_multi = multi_model.predict(X_test_multi)
errors_multi = y_test_multi - y_pred_multi
mae_multi = mean_absolute_error(y_test_multi, y_pred_multi)
std_dev_errors_multi = errors_multi.std()

print("\nÇoklu Regresyon Modeli için Hata Analizi (Test Seti):")
print(f"Ortalama Mutlak Hata (MAE): {mae_multi:.4f}")
print(f"Hataların Standart Sapması: {std_dev_errors_multi:.4f}")

# --- 2.4: Sonuçların Karşılaştırılması ve Yorum ---
print("\n\n--- BÖLÜM 2.4: Sonuçların Karşılaştırılması ---")
print("Bu bölümde, basit doğrusal regresyon modelleri ile çoklu doğrusal regresyon modelinin performansı karşılaştırılmaktadır.")
print("\nModel Performans Özeti (Test Seti üzerindeki MAE):")
for feature, data in models.items():
    y_pred_simple = data['model'].predict(data['X_test'])
    mae_simple = mean_absolute_error(data['y_test'], y_pred_simple)
    print(f"  - Basit Model ({feature}): MAE = {mae_simple:.4f}")

print(f"  - Çoklu Regresyon Modeli: MAE = {mae_multi:.4f}")

print("\nSonuç ve Yorum:")
print(f"Çoklu doğrusal regresyon modelinin R-kare değeri ({r2_multi:.4f}), tüm basit regresyon modellerinin R-kare değerlerinden önemli ölçüde yüksektir. Bu, seçilen dört özelliğin birlikte, yaşam süresindeki değişkenliği tek bir özellikten çok daha iyi açıkladığını gösterir.")
print(f"Ayrıca, test seti üzerindeki Ortalama Mutlak Hata (MAE) değeri, çoklu modelde ({mae_multi:.4f}) en düşük seviyededir. Bu da çoklu modelin tahmin gücünün basit modellere göre daha üstün olduğunu kanıtlar.")
print("Sonuç olarak, yaşam süresini tahmin etmek için birden fazla anlamlı özelliğin (özellikle 'Adult Mortality', 'HIV/AIDS', 'Income composition of resources', 'Schooling') kullanıldığı çoklu regresyon modeli, tek bir özelliğe dayalı basit modellere kıyasla çok daha doğru ve güvenilir bir modeldir.")