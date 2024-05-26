# Telco_Customer_Churn
Developing machine learning models to predict customer retention

# Business Problem
# Telco Customer Churn
**Context**

Dalam bisnis, Customer churn adalah istilah yang digunakan untuk merujuk pada kehilangan pelanggan atau klien dari suatu perusahaan dalam period waktu tertentu.

Dalam hal ini, perusahaan ingin kita dapat memprediksi customer mana yang akan berhenti berlangganan dan mana yang akan terus berlangganan agar perusahaan dapat mendapatkan keuntungan.

Maka dari itu kita akan membuat model machine learning yang akan memprediksi apakah customer akan churn atau tidak, dalam konteks data yang didapatkan yang akan kita buat adalah model machine learning classification.

Target :

- 0 : Tidak akan berhenti berlangganan (Not Churned)
- 1 : Akan berhenti berlangganan (Churned)

**Problem Statement**

Salah satu tantangan terbesar bagi perusahaan di seluruh dunia, termasuk perusahaan telekomunikasi yang memiliki peran besar di masyarakat luas, adalah retensi pelanggan atau lebih tepatnya kehilangan pelanggan. Kerugian yang tinggi yang terjadi akan berdampak pada keuntungan perusahaan dan menghambat pertumbuhan usahanya.

Analisis churn pelanggan merupakan solusi untuk mengurangi kebocoran kehilangan pelanggan. Dari analisis tersebut diperoleh kejelasan mengenai seberapa baik bisnis mempertahankan pelanggan yang merupakan cerminan dari kualitas pelayanan yang diberikan oleh perusahaan sehingga hal ini akan menjadi ukuran yang berguna untuk mengatur strategi produk/jasa dalam menarik pelanggan baru dan menghadapi pesaing bisnis.

**Goals**

Maka dari itu poin poin yang akan menjadi objektif kita antara lain:
- Memprediksi customer mana yang akan churn
- Mencari penyebab apa yang membuat customer churn
- Menghitung keuntungan yang bisa didapat setelah memprediksi churn

**Metrics Evaluation**

Dalam kasus ini kita sangat ingin mempertahankan customer untuk berlangganan sehingga kita ingin meminimalisir False Negative (Customer yang diprediksi tidak Churn tetapi aslinya Churn) dikarenakan dalam konteks retensi customer, lebih mudah mempertahankan customer dibanding dengan mencari customer yang baru, Maka kita akan menggunakan  **Recall** sebagai metriks utama kita.

**Modelling**

Model Machine Learning yang umum digunakan dalam kasus Classification antara lain :
- Logistic Regression
- K Nearest Neighbour
- Decision Tree
- Random Forest
- XGB
- LGBM

#  Conclusion
Model terbaik yang digunakan adalah :
- LGBM dengan learning_rate = 0,2, n_estimators = 50, dan num_leaves = 5
- Undersampler Neighbourhood Cleaning Rule

Setelah melihat classification model LGBM yang kita punya, kita dapat mengambil kesimpulan bahwa apabila kita menggunakan model machine learning kita untuk melihat customer mana yang akan kita coba pertahankan maka model kita dapat mengurangi **83% customer yang akan berhenti berlangganan** untuk segera diberikan strategi-strategi layanan yang diharapkan dapat membuat customer tetap berlangganan, dan model mendapatkan **75% customer yang tetap berlangganan** sehingga dapat kita pelajari pola pola dari customer yang tetap berlangganan ini agar dapat kita terapkan ke customer yang mau berhenti berlangganan.

Model ini memiliki ketepatan dalam memprediksi **customer yang lanjut berlangganan sebanyak 90%**, dan memprediksi **customer yang berhenti berlangganan sebesar 55%**. Penyebab dari model tidak memiliki score precision yang tinggi pada nilai `1`/positif dikarenakan tidak seimbangnya distribusi data dimana nilai positifnya sangat jauh lebih sedikit dibanding nilai negatifnya. 

**Yang berarti bahwa model memiliki keterbatasan disaat memprediksi customer yang akan churn (positif), namun model sangat baik dalam memprediksi customer yang tidak churn (negatif)**.

Hasil model setelah prediksi adalah :
- True Positive (TP) = 213

Adalah customer yang model prediksi dengan benar churn. **Perusahaan akan mengeluarkan biaya untuk mempertahankan customer ini.**

- False Positive (FP) = 177

Adalah customer yang model prediksi dengan salah churn. **Perusahaan mungkin akan mengeluarkan biaya untuk customer ini sebagai pencegahan.**

- True Negative (TN) = 536

Adalah customer yang model prediksi dengan benar tidak churn. Perusahaan tidak perlu mengeluarkan biaya sepeserpun karena customer ini tidak churn.

- False Negative (TN) = 45

Adalah customer yang model prediksi dengan salah tidak churn. Perusahaan dalam hal ini perlu mengeluarkan biaya untuk mencari customer pengganti, dikarenakan customer tipe False Negative (Diprediksi churn namun tidak churn) adalah customer yang 'lolos' dari perhitungan machine learning sehingga mau tidak mau **Perusahaan akan mengeluarkan biaya membuat customer baru berlangganan**


Karena biaya mempertahankan customer itu lebih kecil dibandingkan biaya membuat customer kembali berlangganan, berdasarkan https://www.optimove.com/resources/learning-center/customer-acquisition-vs-retention-costs, maka kita coba asumsikan :

- Biaya mempertahankan customer: $1
- Biaya membuat customer baru berlangganan: $5
- Jumlah Customer : 971 (Berdasarkan jumlah customer yang diprediksi model)


Apabila kita tidak menggunakan model, kita anggap bahwa semua customer akan churn maka :
-  971 x (Biaya mempertahankan customer) = 971 x $1 = $971

Apabila kita menggunakan model :
- (TP x Biaya mempertahankan customer) + (FP x Biaya mempertahankan customer) + (TN x Biaya membuat customer baru berlangganan)
- (213 x $1) + (177 x 1%) + (45 x $5) = $615

Maka oleh dari itu kita menghemat biaya sebesar:
- $971 - $615 = $356

- ($971 - $615) / $971 = 36.7%

Model berhasil menghemat biaya sebanyak $356 dan 36.7% dibanding dengan tidak menggunakan model.

# Recommendation

## Model Recommendation

1. **Model membutuhkan data yang distribusi target variabelnya seimbang**, meskipun model memiliki recall yang tinggi, namun precision yang dimiliki model masih kecil, apalagi saat memprediksi nilai positif. Hal ini diakibatkan tidak seimbangnya distribusi dari target variabel dimana nilai negatif jauh lebih tinggi dibanding nilai positifnya.

2. **Menggunakan model secara langsung**, model dapat digunakan secara langsung/real-time agar dapat memprediksi customer yang akan churn secepat mungkin.

## Business Recommendation
1. **Customer yang baru berlangganan masih banyak yang tidak melanjutkan langganan** hal ini ditunjukkan pada distribusi churn dalam fitur tenure yang menunjukkan bahwa tenure yang kecil cenderung akan churn dibanding tenure yang tinggi, diperlukan adanya perhatian khusus kepada customer yang baru berlangganan, bisa dengan diberikan bonus bonus layanan untuk pengguna baru atau semacamnya.

2. **Buat promosi yang menarik** Customer yang diprediksi machine learning akan churn agar diberikan promosi yang menarik untuk mencegah customer untuk berhenti berlangganan.

3. **Customer yang contractnya month-to-month cenderung churn dibanding yang yearly**, sistem contract jangka panjang mungkin akan mencegah customer untuk churn, sehingga kita dapat menghapus sistem contract month-to-month, namun perlu diperhatikan bahwa hal ini bisa malah berdampak negatif dalam menarik customer dikarenakan customer akan lebih berhati-hati dalam berlangganan langsung dalam jangka panjang.
