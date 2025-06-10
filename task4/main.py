import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# --- GÖREV 1: Veri Setini Yükleme ---
print("--- GÖREV 1: MNIST veri seti indiriliyor... (Bu işlem birkaç dakika sürebilir)")
start_time = time.time()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
end_time = time.time()
print(f"Veri seti {end_time - start_time:.2f} saniyede indirildi.")

# --- GÖREV 2 & 3: Veri Yapısını İnceleme ve Loglama ---
print("\n--- GÖREV 2 & 3: Veri Yapısının İncelenmesi ---")
X = mnist.data
y = mnist.target

num_samples, num_features = X.shape
unique_labels = np.unique(y)
log_info = (
    f"MNIST Veri Seti Analizi\n"
    f"-------------------------\n"
    f"Toplam Örnek (Sample) Sayısı: {num_samples}\n"
    f"Özellik (Feature) Sayısı her örnek için: {num_features}\n"
    f"Veri (X) matrisinin boyutu: {X.shape}\n"
    f"Etiket (y) vektörünün boyutu: {y.shape}\n"
    f"Etiketler (Sınıflar): {unique_labels}\n"
    f"Toplam Sınıf Sayısı: {len(unique_labels)}\n"
)
print(log_info)

# Bilgileri bir log dosyasına kaydet
log_filename = "mnist_data_log.txt"
# HATA DÜZELTMESİ: Dosyayı evrensel UTF-8 kodlamasıyla açıyoruz.
with open(log_filename, "w", encoding="utf-8") as f:
    f.write(log_info)
print(f"Veri bilgileri '{log_filename}' dosyasına kaydedildi.")


# --- GÖREV 4: Örnek Rakamları Görselleştirme ---
print("\n--- GÖREV 4: Örnek Rakamların Gösterimi ---")
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()
for i in range(10):
    idx = np.random.randint(0, num_samples)
    img = X[idx].reshape(28, 28)
    axes[i].imshow(img, cmap='gray_r')
    axes[i].set_title(f"Etiket: {y[idx]}")
    axes[i].axis('off')
plt.suptitle('MNIST Veri Setinden Rastgele Örnekler', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- GÖREV 5: Veriyi Eğitim ve Test Olarak Ayırma ---
print("\n--- GÖREV 5: Verinin Bölünmesi ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
print(f"Test seti boyutu: {X_test.shape[0]} örnek")


# --- GÖREV 6: Naive Bayes Modeli Seçimi ve Eğitimi ---
print("\n--- GÖREV 6: Model Seçimi ve Eğitimi ---")
print("Model Seçimi: Gaussian Naive Bayes (GaussianNB)")
print("Gerekçe: MNIST veri setindeki özellikler (pikseller) 0-255 arasında değişen sürekli değerlerdir.")
print("Bu tür sürekli veriler için en uygun Naive Bayes türü GaussianNB'dir, çünkü her bir özelliğin dağılımının Gauss (Normal) dağılımı olduğunu varsayar.")

gnb = GaussianNB()
print("\nModel eğitiliyor...")
start_time = time.time()
gnb.fit(X_train, y_train)
end_time = time.time()
print(f"Model {end_time - start_time:.2f} saniyede eğitildi.")


# --- GÖREV 7: Karışıklık Matrisi (Confusion Matrix) ---
print("\n--- GÖREV 7: Karışıklık Matrisinin Oluşturulması ve Analizi ---")
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=gnb.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Naive Bayes Sınıflandırıcısı Karışıklık Matrisi')
plt.show()

print("\nKarışıklık Matrisi Yorumu:")
print(" - Matrisin köşegeni doğru sınıflandırılan örneklerin sayısını gösterir. Bu değerlerin yüksek olması beklenir.")
print(" - Köşegen dışındaki hücreler ise yanlış sınıflandırmaları gösterir. Örneğin, modelin en çok karıştırdığı rakam çiftleri genellikle görsel olarak birbirine benzeyenlerdir (4-9, 5-3, 8-3 gibi).")


# --- GÖREV 8: Doğruluk ve Diğer Metrikler ---
print("\n--- GÖREV 8: Performans Metrikleri ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin Genel Doğruluk Oranı: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nSınıf Bazında Detaylı Rapor:")
report = classification_report(y_test, y_pred)
print(report)

print("Her Sınıf İçin Hata Oranı:")
class_error_rate = 1 - cm.diagonal() / cm.sum(axis=1)
for i, label in enumerate(gnb.classes_):
    print(f"  - Sınıf '{label}': Hata Oranı = {class_error_rate[i]:.3f} ({class_error_rate[i]*100:.1f}%)")

print("\nDiğer Faydalı Metrikler ve Yorum:")
print(" - Precision (Kesinlik): Bir sınıf için yapılan tahminlerin ne kadarının doğru olduğunu gösterir.")
print(" - Recall (Duyarlılık): Bir sınıfa ait olan örneklerin ne kadarının doğru tahmin edildiğini gösterir.")
print(" - F1-Skoru: Precision ve Recall'un harmonik ortalamasıdır, iki metriği dengeleyerek genel bir başarı ölçütü sunar.")

