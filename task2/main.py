import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# --- GÖREV 1 & 2: Veri Yükleme ve Özellik/Etiket Erişimi ---
print("--- GÖREV 1 & 2: Veri Yükleme ve Hazırlama ---")
iris = load_iris()
X = iris.data  # Özellikler (features)
y = iris.target  # Etiketler (labels)
print(f"Veri seti yüklendi. Özellik (X) matrisi boyutu: {X.shape}, Etiket (y) vektörü boyutu: {y.shape}\n")


# --- GÖREV 3: Veriyi Eğitim ve Test Olarak Ayırma ---
# Başlangıç için veriyi %75 eğitim, %25 test olarak ayıralım.
# random_state, her çalıştırmada aynı ayırımın yapılmasını sağlar.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("--- GÖREV 3: Veri Ayırma ---")
print(f"Eğitim seti boyutu: {X_train.shape[0]} kayıt")
print(f"Test seti boyutu: {X_test.shape[0]} kayıt\n")


# --- GÖREV 4: Karar Ağacı Modelini Eğitme ---
print("--- GÖREV 4: Modeli Eğitme ---")
# Modeli oluşturma
decision_tree_model = DecisionTreeClassifier(random_state=42)
# Modeli eğitim verisiyle eğitme
decision_tree_model.fit(X_train, y_train)
print("Karar Ağacı modeli başarıyla eğitildi.\n")


# --- GÖREV 5 & 6: Eğitim ve Test Verileri için Tahmin ---
print("--- GÖREV 5 & 6: Tahmin Yapma ---")
# Eğitim verisi üzerinde tahmin yapma
y_pred_train = decision_tree_model.predict(X_train)
# Test verisi üzerinde tahmin yapma
y_pred_test = decision_tree_model.predict(X_test)
print("Eğitim ve test verileri için tahminler yapıldı.\n")


# --- GÖREV 7: Doğruluk Skorlarını Hesaplama ---
print("--- GÖREV 7: Doğruluk Skorları ---")
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Eğitim Verisi Doğruluk Skoru: {accuracy_train:.4f}")
print(f"Test Verisi Doğruluk Skoru: {accuracy_test:.4f}\n")


# --- GÖREV 8: Eğitim Verisi Sonucunun Anlamı ---
print("--- GÖREV 8: Eğitim Verisi Sonucunun Yorumu ---")
print("Eğitim verisi üzerindeki doğruluk skorunun 1.0 (%100) olması, modelin eğitim setindeki tüm örnekleri mükemmel bir şekilde öğrendiğini gösterir.")
print("Bu durum, modelin eğitim verisini 'ezberlediği' anlamına gelebilir ve 'aşırı öğrenme' (overfitting) olarak adlandırılır.")
print("Aşırı öğrenen bir model, daha önce görmediği yeni verilerde (test verisi gibi) genellikle daha düşük performans gösterir. Bu yüzden test verisi doğruluğu, modelin gerçek dünya performansını ölçmek için daha önemli bir metriktir.\n")


# --- GÖREV 9: Farklı random_state Değerlerinin Etkisi ---
print("--- GÖREV 9: random_state Değişiminin İncelenmesi ---")
test_accuracies = []
# 10 farklı random_state değeri için deneme yap
for i in range(10):
    # Veriyi her seferinde farklı bir random_state ile böl
    X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(X, y, test_size=0.25, random_state=i)
    # Modeli eğit
    model_rs = DecisionTreeClassifier(random_state=42) # Modelin kendi random_state'ini sabit tutuyoruz
    model_rs.fit(X_train_rs, y_train_rs)
    # Test verisi üzerinde tahmin yap ve doğruluğu kaydet
    y_pred_rs = model_rs.predict(X_test_rs)
    accuracy_rs = accuracy_score(y_test_rs, y_pred_rs)
    test_accuracies.append(accuracy_rs)
    print(f"random_state={i}, Test Doğruluğu: {accuracy_rs:.4f}")

# Ortalama ve standart sapmayı hesapla
mean_accuracy = np.mean(test_accuracies)
std_dev_accuracy = np.std(test_accuracies)
print(f"\n10 farklı deneme sonucu:")
print(f"Ortalama Test Doğruluğu: {mean_accuracy:.4f}")
print(f"Test Doğruluğu Standart Sapması: {std_dev_accuracy:.4f}\n")


# --- GÖREV 10 & 11: Farklı Veri Bölme Oranlarının Test Edilmesi ve Grafiği ---
print("--- GÖREV 10 & 11: Farklı Bölme Oranlarının Testi ---")
split_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
accuracy_by_ratio = []

for ratio in split_ratios:
    # Veriyi mevcut orana göre böl (1-ratio eğitim, ratio test)
    X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(X, y, test_size=ratio, random_state=42)
    
    # Boş eğitim seti oluşmasını engelle
    if len(y_train_ratio) == 0:
        print(f"Test oranı {ratio} için eğitim verisi kalmadı, atlanıyor.")
        accuracy_by_ratio.append(np.nan) # Grafikte boşluk için
        continue

    model_ratio = DecisionTreeClassifier(random_state=42)
    model_ratio.fit(X_train_ratio, y_train_ratio)
    
    y_pred_ratio = model_ratio.predict(X_test_ratio)
    acc_ratio = accuracy_score(y_test_ratio, y_pred_ratio)
    accuracy_by_ratio.append(acc_ratio)
    print(f"Test Oranı: {ratio}, Test Doğruluğu: {acc_ratio:.4f}")

# Doğruluk skorlarını grafiğe dök
plt.figure(figsize=(10, 6))
plt.plot(split_ratios, accuracy_by_ratio, marker='o', linestyle='--')
plt.title('Test Verisi Oranına Göre Model Doğruluğu')
plt.xlabel('Test Verisi Oranı (test_size)')
plt.ylabel('Doğruluk Skoru (Accuracy)')
plt.xticks(split_ratios)
plt.grid(True)
plt.show()


# --- GÖREV 12: Örnek Bir Karar Ağacını Çizdirme ---
print("\n--- GÖREV 12: Örnek Karar Ağacı Çizimi ---")
print("Aşağıdaki grafikte, ilk başta eğittiğimiz modelin görselleştirilmiş hali bulunmaktadır.")
plt.figure(figsize=(20, 15))
plot_tree(decision_tree_model, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Eğitilmiş Karar Ağacı Modeli")
plt.show()

