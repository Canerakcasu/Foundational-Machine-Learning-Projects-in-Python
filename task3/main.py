import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import sys

# --- GÖREV 1 & 2: GÜNCELLENMİŞ ve DAHA SAĞLAM Parser Fonksiyonu ---
def parse_data(file_name):
    """
    Verilen dosya adını, çalışan script ile aynı klasörde arar,
    bozuk satırları temizler ve bir pandas DataFrame'e dönüştürür.
    """
    try:
        # Script'in bulunduğu klasörün tam yolunu al
        # Bu satır, script'i .py dosyası olarak çalıştırırken işe yarar.
        # Eğer bir notebook'ta (Jupyter, Colab) çalıştırıyorsanız,
        # dosya adlarını tam yoluyla yazmanız gerekebilir.
        try:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_directory, file_name)
        except NameError:
            # __file__ tanımlı değilse (örneğin interaktif bir shell'de),
            # mevcut çalışma dizinini kullan.
            full_path = file_name
            
        # Dosyayı oku, ayırıcı olarak herhangi bir boşluk karakterini kabul et
        df = pd.read_csv(full_path, sep=r'\s+', header=None, engine='python', on_bad_lines='warn')
        
        # Sütun sayısına göre isimlendirme yap
        if df.shape[1] >= 3:
            df = df.iloc[:, [0, 1, 2]] # Sadece ilk 3 sütunu al
            df.columns = ['x', 'y', 'true_cluster']
        elif df.shape[1] == 2:
            df.columns = ['x', 'y']
        else:
            print(f"HATA: '{file_name}' dosyasında yetersiz sütun sayısı.")
            return None
        
        # --- BOZUK VERİYİ TEMİZLEME LOGIĞİ ---
        # Sayısal olmayan değerleri NaN'a dönüştür (Not a Number)
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Temizleme öncesi satır sayısını kaydet
        original_rows = len(df)
        
        # Sayısal olmayan satırları (NaN içerenleri) veri setinden kaldır
        df.dropna(subset=['x', 'y'], inplace=True)
        
        # Kaç satırın atıldığını kontrol et ve bildir
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            print(f"UYARI: '{file_name}' dosyasından {dropped_rows} adet sayısal olmayan satır atlandı.")
        # --- BİTTİ ---

        if 'true_cluster' in df.columns:
            # true_cluster sütununu da sayısal yap ve tamsayıya çevir
            df['true_cluster'] = pd.to_numeric(df['true_cluster'], errors='coerce').fillna(0).astype(int)

        print(f"'{file_name}' dosyası başarıyla okundu ve temizlendi. Boyut: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"HATA: '{file_name}' dosyası bulunamadı.")
        print(f"Aranan tam yol: {os.path.abspath(file_name)}")
        print("Lütfen dosyanın script ile aynı klasörde olduğundan ve adının doğru yazıldığından emin olun.")
        return None
    except Exception as e:
        print(f"'{file_name}' okunurken bir hata oluştu: {e}")
        return None

# --- GÖREV 3: Veri Setleri Arasındaki Farklılıkların Analizi ---
print("--- GÖREV 3: Veri Setlerinin İncelenmesi ve Kümelemeye Etkileri ---\n")

file_paths_s = ['s1.txt', 's2.txt', 's3.txt', 's4.txt']
data_sets_s = {path: parse_data(path) for path in file_paths_s}

# Veri setlerini görselleştirme
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
titles = ['s1: 15 Yoğun Küme', 's2: 4 Yoğun Küme', 's3: Gürültülü 15 Küme', 's4: Uzunlamasına Kümeler']

for i, (path, df) in enumerate(data_sets_s.items()):
    if df is not None:
        axs[i].scatter(df['x'], df['y'], s=8)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('X Ekseni')
        axs[i].set_ylabel('Y Ekseni')
plt.suptitle('Kümeleme Öncesi Veri Setlerinin Görsel Analizi', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nVeri Setlerinin Kümeleme Sürecine Etkileri:")
print(" - s1.txt ve s2.txt: Bu veri setleri, küresel (yuvarlak) ve birbirinden net bir şekilde ayrılmış kümeler içeriyor. K-Means algoritmasının bu tür 'ideal' yapılarda çok başarılı olması beklenir.")
print(" - s3.txt: Bu set, s1'e benziyor ancak kümelerin etrafında daha fazla 'gürültü' (outlier) veri noktası içeriyor. K-Means, bu gürültü noktalarını da bir kümeye atamaya çalışacağı için küme merkezlerinin (centroid) yerini bir miktar kaydırabilir.")
print(" - s4.txt: Bu set, küresel olmayan (uzunlamasına) kümelere sahiptir. K-Means, kümelerin küresel olduğunu varsaydığı ve varyansı minimize etmeye çalıştığı için bu tür yapıları doğru bir şekilde ayıramaz. Algoritmanın bu veri setinde zorlanması beklenir.\n")


# --- GÖREV 4 & 5: S1-S4 için K-Means Uygulaması ve Analizi ---
print("\n--- GÖREV 4 & 5: S-Serisi Veri Setleri için K-Means Kümelemesi ---")

k_values = {'s1.txt': 15, 's2.txt': 4, 's3.txt': 15, 's4.txt': 2}

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

for i, (path, df) in enumerate(data_sets_s.items()):
    if df is not None:
        X = df[['x', 'y']].values
        k = k_values[path]
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        y_kmeans = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        ax = axs[i]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap='viridis', alpha=0.8)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')
        ax.set_title(f'{path} için K-Means (k={k})')
        ax.set_xlabel('X Ekseni')
        ax.set_ylabel('Y Ekseni')
        ax.legend()

plt.suptitle('K-Means Kümeleme Sonuçları', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nSonuçların Analizi (İnsan Gözüyle):")
print(" - s1 ve s2: K-Means, beklendiği gibi, bu veri setlerindeki küresel ve ayrık kümeleri mükemmel bir şekilde tespit etti. Merkezler, kümelerin tam ortasına yerleşti.")
print(" - s3: Algoritma, gürültüye rağmen ana kümeleri büyük ölçüde doğru buldu. Ancak bazı aykırı noktalar, merkezlerin konumunu hafifçe etkilemiş olabilir. Yine de sonuç oldukça başarılı.")
print(" - s4: K-Means'in sınırlılıkları bu örnekte açıkça görülüyor. Algoritma, uzunlamasına yapıları ayıramadı ve veriyi küresel varsayımına göre ikiye böldü. Bu, veri yapısına uygun olmayan bir sonuçtur ve K-Means'in her veri tipi için uygun olmadığını gösterir.")


# --- GÖREV 6: Spiral Veri Seti Analizi ---
print("\n\n--- GÖREV 6: Spiral Veri Seti Analizi ---")
spiral_df = parse_data('spiral.txt')

if spiral_df is not None:
    X_spiral = spiral_df[['x', 'y']].values
    y_true_spiral = spiral_df['true_cluster'].values
    
    kmeans_spiral = KMeans(n_clusters=3, random_state=42, n_init='auto')
    y_kmeans_spiral = kmeans_spiral.fit_predict(X_spiral)
    centroids_spiral = kmeans_spiral.cluster_centers_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_kmeans_spiral, s=15, cmap='viridis')
    ax1.scatter(centroids_spiral[:, 0], centroids_spiral[:, 1], c='red', s=150, marker='X')
    ax1.set_title('K-Means Kümeleme Sonucu (spiral.txt)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_true_spiral, s=15, cmap='viridis')
    ax2.set_title('Gerçek Küme Yapısı (spiral.txt)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.suptitle('K-Means vs. Gerçek Küme Karşılaştırması', fontsize=16)
    plt.show()
    
    print("\nKarşılaştırma ve Açıklama:")
    print(" - Sol taraftaki grafik, K-Means'in 'spiral' verisini nasıl kümelediğini gösterir. Görüldüğü gibi, algoritma veriyi basitçe üç parçaya bölmüştür. Bunun nedeni, K-Means'in noktaların merkezlere olan uzaklığına göre kümeleme yapması ve kümelerin dışbükey (convex) ve küresel (isotropic) olduğunu varsaymasıdır.")
    print(" - Sağ taraftaki grafik ise verinin gerçek yapısını göstermektedir. Veri, iç içe geçmiş üç adet spiral koldan oluşmaktadır.")
    print(" - Sonuç: K-Means, bu tür iç içe geçmiş ve küresel olmayan karmaşık yapıları tespit etmek için uygun bir algoritma değildir. Bu tür veri setleri için DBSCAN veya Spektral Kümeleme gibi yoğunluk veya bağlantı bazlı algoritmalar çok daha iyi sonuçlar verir.")

