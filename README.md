# Predictive Analytics Pada Car Evaluation Dataset

## Domain Proyek

Industri asuransi mobil merupakan salah satu industri yang penting dalam perekonomian. Asuransi mobil memberikan perlindungan kepada pemilik kendaraan dari kerugian finansial akibat kecelakaan, kerusakan, atau pencurian. Premi asuransi mobil merupakan sumber pendapatan utama bagi perusahaan asuransi.

Pada industri asuransi, salah satu faktor penting dalam menentukan premi asuransi mobil adalah tingkat risiko mobil tersebut. Tingkat risiko ini biasanya ditentukan berdasarkan berbagai faktor, seperti jenis mobil, usia mobil, riwayat kecelakaan, dan lain sebagainya. Proses penetapan premi asuransi secara manual dapat memakan waktu dan rawan terhadap kesalahan. Hal ini dapat menyebabkan inefisiensi dan ketidakakuratan dalam penetapan premi asuransi.

Oleh karena itu, dibutuhkan solusi yang dapat membantu perusahaan asuransi dalam menentukan premi asuransi secara lebih akurat dan efisien. Salah satu solusi yang potensial adalah dengan menggunakan teknik Predictive Analytics untuk membangun model prediksi peringkat risiko mobil.

**Referensi: [AIRA-ML: Auto Insurance Risk AssessmentMachine Learning Model using Resampling Methods](https://thesai.org/Downloads/Volume14No9/Paper_66-AIRA_ML_Auto_Insurance_Risk_Assessment_Machine_Learning_Model.pdf)** 

## Business Understanding

### Problem Statements

- Algoritma Predictive Analytics apa yang paling cocok untuk membangun model prediksi peringkat risiko mobil?
- Bagaimana menerapkan model prediksi peringkat risiko mobil dalam proses penetapan premi asuransi mobil serta cara mengevaluasi akurasi dan efisiensi model prediksi tersebut?

### Goals

- Membangun model prediksi peringkat risiko mobil yang akurat dan efisien.
- Meningkatkan akurasi dan efisiensi dalam penetapan premi asuransi mobil.

### Solution statements
- Menggunakan 3 algoritma regresi pada pembangunan model prediksi antara lain adalah Linear Regression, Decision Tree, dan Random Forest. 

- Ketiga algoritma regresi yang telah dijabarkan akan dilakukan evaluasi dengan menggunakan metrik evaluasi Mean Squared Error (MSE). Setelah itu, akan dilakukan perbandingan nilai MSE pada ketiga algoritma regresi yang telah dilatih untuk mengambil model algoritma yang terbaik dimana perbandingan tersebut dilakukan dengan cara mengambil nilai MSE yang paling kecil yang menunjukkan model yang paling akurat.

## Data Understanding
Data yang digunakan pada proyek ini diambil dari data evaluasi mobil yang berasal dari website UCI Machine Learning dimana pada dataset ini berisikan informasi tentang 1728 mobil beserta 6 fiturnya, seperti harga beli, biaya perawatan, jumlah pintu, perkiraan keselamatan, dan sebagainya. Fitur-fitur pada dataset ini merupakan tipe fitur kategorikal dimana penamaan setiap fitur menggunakan label masing-masing. Berikut adalah sumber dataset yang digunakan pada proyek ini:

Dataset: [Car Evaluation Data Set](https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set).

Referensi Tambahan Dataset: [Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)

### Variabel-variabel pada Car Evaluation dataset adalah sebagai berikut:
- buying : merupakan harga beli pada mobil.
- maint : merupakan harga perawatan pada mobil.
- doors : merupakan jumlah pintu yang terdapat pada mobil.
- persons : merupakan jumlah kapasitas orang yang dapat ditampung mobil.
- lug_boot : merupakan ukuran bagasi mobil.
- safety : merupakan perkiraan keamanan mobil.
- class : merupakan tingkat evaluasi pada mobil untuk menentukan premi asuransi mobil.

### Exploratory Data Analysis (EDA)
- Melakukan rename kolom pada dataset sesuai dengan variabel-variabel yang ada pada Car Evaluation Dataset.
- Menampilkan dataframe yang telah direname kolomnya.

Output: 
|index|buying|maint|doors|persons|lug\_boot|safety|class|
|---|---|---|---|---|---|---|---|
|0|vhigh|vhigh|2|2|small|med|unacc|
|1|vhigh|vhigh|2|2|small|high|unacc|
|2|vhigh|vhigh|2|2|med|low|unacc|
|3|vhigh|vhigh|2|2|med|med|unacc|
|4|vhigh|vhigh|2|2|med|high|unacc|

- Mengecek informasi pada dataset berupa banyak variabel, nama-nama variabel, dan tipe data pada tiap variabel. 

        Output: 
        RangeIndex: 1727 entries, 0 to 1726
        Data columns (total 7 columns):
        #   Column    Non-Null Count  Dtype 
        ---  ------    --------------  ----- 
        0   buying    1727 non-null   object
        1   maint     1727 non-null   object
        2   doors     1727 non-null   object
        3   persons   1727 non-null   object
        4   lug_boot  1727 non-null   object
        5   safety    1727 non-null   object
        6   class     1727 non-null   object
        dtypes: object(7)

- Mengecek deskripsi statistik pada dataset.

Output:
|index|buying|maint|doors|persons|lug\_boot|safety|class|
|---|---|---|---|---|---|---|---|
|count|1727|1727|1727|1727|1727|1727|1727|
|unique|4|4|4|3|3|3|4|
|top|high|high|3|4|med|med|unacc|
|freq|432|432|432|576|576|576|1209|

- Mengeksplorasi variabel target yang adalah variabel **'class'**.
        
        Output:
        class
        unacc    1209
        acc       384
        good       69
        vgood      65
        Name: count, dtype: int64

- Mengecek missing value (nilai null pada dataset).

        Output:
        buying      0
        maint       0
        doors       0
        persons     0
        lug_boot    0
        safety      0
        class       0
        dtype: int64

- Menganalisis jumlah data pada setiap variabel yang ada dalam dataset.

        Output:
        buying
        high     432
        med      432
        low      432
        vhigh    431
        Name: count, dtype: int64
        maint
        high     432
        med      432
        low      432
        vhigh    431
        Name: count, dtype: int64
        doors
        3        432
        4        432
        5more    432
        2        431
        Name: count, dtype: int64
        persons
        4       576
        more    576
        2       575
        Name: count, dtype: int64
        lug_boot
        med      576
        big      576
        small    575
        Name: count, dtype: int64
        safety
        med     576
        high    576
        low     575
        Name: count, dtype: int64
        class
        unacc    1209
        acc       384
        good       69
        vgood      65
        Name: count, dtype: int64

## Data Preparation
- Train-Test Split : 
    - Data preparation dimulai dengan membagi dataset menjadi data train dan data test dengan jumlah rasio 80% pada data train dan 20% pada data test. 
    - Alasan dilakukannya pembagian data train dan data test adalah untuk melatih model machine learning dan mengevaluasi kinerja model.
- Encoding Fitur Kategori : 

    - Tahap ini dimulai dengan mengimpor library **"sklearn.preprocessing.OrdinalEncoder"** yang digunakan untuk melakukan ordinal encoding. 
            
            # Import kategori encoder
            !pip install category_encoders
            import category_encoders as ce
       

    - Melakukan inisialisasi OrdinalEncoder dengan parameter *'cols'* dimana parameter ini berisi daftar kolom variabel fitur pada dataset yang ingin di-encode. Kolom variabel fitur tersebut antara lain *'buying', 'maint', 'doors', 'persons', 'lug_boot', dan 'safety'*.
            
            # Encode fitur-fitur dengan ordinal encoding
            encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    
    - Selanjutnya, fit dan transform data train dan data test dimana fit digunakan untuk melatih OrdinalEncoder pada data train (X_train) untuk mempelajari urutan (ordinal) dari nilai-nilai unik dalam setiap kolom yang ditentukan. Setelah OrdinalEncoder terlatih pada data train, metode transform digunakan untuk mengkonversi nilai-nilai dalam data test (X_test).

            X_train = encoder.fit_transform(X_train)
            X_test = encoder.transform(X_test)

    - Alasan dilakukannya encoding fitur kategori adalah untuk membantu membuat model lebih mudah untuk diinterpretasikan dengan menyediakan representasi numerik yang lebih mudah dipahami.

## Modeling

Pada tahap ini, proyek ini menerapkan 3 model development antara lain sebagai berikut:

- **Model Development dengan Linear Regression**      
- **Model Development dengan Decision Tree**
- **Model Development dengan Random Forest**

#### **Tahapan dan Parameter Pada Setiap Algoritma**

- **Preprocessing (Berlaku untuk semua model):**
        
        1. Import library dari scikit-learn.
        2. Label encoding : Langkah ini diperlukan untuk mengubah variabel target y_train berisi nilai kategorikal (seperti label teks) dimana LabelEncoder mengubah label ini menjadi nilai numerik agar model dapat memahaminya.

- **Pelatihan Model (Spesifik untuk setiap algoritma):**

  **a. Linear Regression :**

        - Pembuatan Model: Kode membuat objek LinearRegression (LR). Model ini mengasumsikan hubungan linier antara fitur dalam X_train dan variabel target y_train_encoded (setelah kemungkinan pengkodean).

        - Pemasangan Model: Model dilatih menggunakan metode fit. Ini mengambil data pelatihan X_train (fitur) dan y_train_encoded (variabel target) sebagai input dan mempelajari koefisien persamaan linier yang paling sesuai dengan data.

  **b. Decision Tree :**

        - Pembuatan Model: Kode membuat objek DecisionTreeRegressor (DT) dengan konfigurasi tertentu:
          1. max_depth=16: Ini membatasi kedalaman maksimum pohon keputusan, mencegah overfitting.
          2. random_state=55: Ini menetapkan benih untuk keacakan, memastikan reproduktifitas hasil.

        - Pemasangan Model: Mirip dengan Linear Regression, metode fit digunakan untuk melatih model pada X_train dan y_train_encoded.

  **c. Random Forest :**

        - Pembuatan Model: Kode membuat objek RandomForestRegressor (RF) dengan konfigurasi tertentu:
          1. n_estimators=50: Ini menetapkan jumlah pohon keputusan yang akan ditanam di hutan (50 dalam kasus ini).
          2. max_depth=16: Mirip dengan pohon keputusan, ini membatasi kedalaman setiap pohon individu.
          3. random_state=55: Memastikan reproduktifitas.
          4. n_jobs=-1: Ini menggunakan semua core yang tersedia di mesin Anda untuk pelatihan yang lebih cepat.

        - Pemasangan Model: Metode fit melatih Random Forest. Ini menciptakan beberapa decision tree, masing-masing dengan kumpulan fitur dan titik data acak. Prediksi dari pohon-pohon ini kemudian dirata-ratakan untuk prediksi yang lebih kuat.

#### Kelebihan & Kekurangan Masing-Masing Algoritma
        
- **Regresi Linear**
        
  **Kelebihan:**

         a. Mudah dipahami dan diimplementasikan. Persamaan modelnya mudah diinterpretasikan, menunjukkan hubungan antara fitur dan variabel target.
         b. Perhitungan yang efisien. Melatih model regresi linear lebih murah secara komputasi dibandingkan dengan algoritma lain.
         c. Membentuk dasar yang baik untuk metode lain. Regresi linear adalah dasar untuk banyak teknik statistik lainnya.

  **Kekurangan:**

         a. Memerlukan data linear. Mengasumsikan hubungan linear antara fitur dan variabel target.
         b. Sensitif terhadap outlier. Outlier dapat secara signifikan memengaruhi kinerja model.
         c. Tidak cocok untuk masalah kompleks. Mungkin tidak menangkap hubungan kompleks dalam data.

- **Decision Tree**
        
  **Kelebihan:**

         a. Mudah dipahami dan diinterpretasikan. Struktur pohon keputusan mudah divisualisasikan dan dipahami logika di balik prediksi.
         b. Tidak memerlukan penskalaan fitur. Pohon keputusan tidak peka terhadap penskalaan fitur.
         c. Dapat menangani data dengan nilai yang hilang. Dapat menangani nilai yang hilang dalam data tanpa memerlukan imputasi.

  **Kekurangan:**

         a. Rentan terhadap overfitting. Pohon keputusan dapat dengan mudah overfit data pelatihan, yang mengarah pada kinerja yang buruk pada data yang tidak terlihat.
         b. Varians tinggi. Perubahan kecil dalam data pelatihan dapat menyebabkan perubahan besar dalam struktur pohon keputusan.
         c. Kurang akurat untuk masalah kontinu. Pohon keputusan mungkin tidak seakurat algoritma lain untuk variabel target kontinu.

- **Random Forest**

  **Kelebihan:**

         a. Akurasi tinggi. Hutan acak dapat mencapai akurasi tinggi pada berbagai masalah.
         b. Tahan terhadap overfitting. Dengan merata-ratakan prediksi dari beberapa pohon keputusan, hutan acak kurang rentan terhadap overfitting.
         c. Dapat menangani data dengan nilai yang hilang. Dapat menangani nilai yang hilang dalam data tanpa memerlukan imputasi.
        
  **Kekurangan:**

         a. Kotak hitam. Cara kerja internal hutan acak bisa sulit untuk diinterpretasikan.
         b. Perhitungan lebih mahal. Melatih hutan acak bisa lebih mahal secara komputasi daripada melatih satu pohon keputusan.
         c. Memerlukan penyesuaian parameter. Menyetel hyperparameter seperti jumlah pohon dan kedalaman maksimum bisa memakan waktu.

#### Pemilihan Model Terbaik dari Ketiga Algoritma Model
- **Model Terbaik** : Model Development dengan Random Forest

- **Alasan** : Karena pada tahap evaluasi model, model random forest memiliki nilai MSE yang paling kecil dimana hal ini menunjukkan bahwa model **Random Forest** adalah model yang paling akurat dibanding kedua model lainnya. 

## Evaluation

Tahap evaluasi model pada proyek Predictive Analytics ini menggunakan metrik Mean Squared Error (MSE). 

#### Ringkasan Hasil 
Tabel mse menunjukkan Mean Squared Error (MSE) untuk setiap model (Linear Regression, Decision Tree, dan Random Forest) pada data train dan test. Semakin rendah nilai MSE, semakin baik kinerja model. 


                                train	    test
        LinearRegression	0.000608	0.000687
        DecisionTree	         0.0	0.000145
        RandomForest	    0.000011	0.000097

Berdasarkan MSE, Random Forest memiliki kinerja terbaik pada data train dan test, diikuti oleh Decision Tree dan Linear Regression.

#### Penjelasan Metrik Evaluasi

Mean Squared Error (MSE):

MSE adalah metrik evaluasi yang umum digunakan untuk mengukur kesalahan prediksi model regresi. MSE dihitung dengan mengambil rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Formula MSE adalah:

![MSE](https://github.com/user-attachments/assets/9e950e02-3753-4c42-abf6-b1d14a9670c8)

Dimana:

n adalah jumlah data

y_pred_i adalah nilai prediksi ke-i

y_true_i adalah nilai aktual ke-i

**Cara Kerja MSE:**

MSE mengukur rata-rata kesalahan prediksi model. Semakin kecil nilai MSE, semakin dekat nilai prediksi model dengan nilai aktual. Nilai MSE 0 menunjukkan bahwa model memprediksi semua nilai dengan benar.

**Interpretasi MSE:**

Nilai MSE yang "kecil" dapat didefinisikan secara berbeda tergantung pada skala data dan kompleksitas model. Dalam proyek ini, nilai MSE dibagi dengan 1e3 untuk memudahkan interpretasi.

Pengujian Model dengan Model Terbaik (Random Forest)

        # Pengujian model dengan prediksi class dari data test
        predict = X_test.iloc[:5].copy()
        pred_dict = {'y_true':y_test[:5]}
        for name, model in model_dict.items():
        pred_dict['prediksi_'+name] = model.predict(predict).round(2)

        pd.DataFrame(pred_dict)


Output: 

![Output](https://github.com/user-attachments/assets/20754314-38aa-425c-a131-78b675a17192)

#### Kesimpulan
*Dengan adanya perbandingan setiap algoritma model menggunakan MSE, ini dapat menjawab problem statements dan goals yang telah dijabarkan sebelumnya serta membuktikan bahwa solution statements yang diberikan berdampak pada perusahaan asuransi dalam menentukan premi asuransi secara lebih akurat dan efisien dimana perbandingan algoritma model menggunakan MSE ini menunjukkan bahwa model algoritma Predictive Analystics yang paling cocok untuk prediksi peringkat risiko mobil adalah **Random Forest**. Untuk itu, model terbaik ini dapat di implementasikan dengan mengintegrasikan model prediksi terbaik (Random Forest) dengan sistem data asuransi yang ada untuk memprediksi peringkat risiko mobil secara akurat.*
