import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from sklearn import metrics


def euclidean_distance(x1, x2):#hàm tính khoảng cach euclidean
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:


    def __init__(self, K=5, max_iters=100000):
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)] #tạo ra số List lưu thông tin của từng cum theo số K đầu vào
     
        self.centroids = [] #tạo danh sách các tâm cụm

    def fit(self,X):# khi thực hiện xong các centroid đã được xác định dựa vào số K
        self.X = X
        self.n_samples, self.n_features = X.shape 

     
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False) #lấy ngẫu nhiên K mẫu làm K điểm tâm cụm khởi tạo
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
       
        for _ in range(self.max_iters): #lặp tối đa max_iters lần
           
            self.clusters = self._create_clusters(self.centroids) #Phân cụm cho từng điểm dữ liệu dựa vào các tâm cụm đã có
            centroids_old = self.centroids #Lưu lại tâm cụm hiện tại
            self.centroids = self._get_centroids(self.clusters)#sau khi phân cụm cho tất cả điểm dữ liệu, tính lại các tâm cụm mới dựa vào trung bình cộng tất cả các điểm có trong cụm đó
            if self._is_converged(centroids_old, self.centroids): #Nếu tâm cụm mới không khác so với tâm cụm hiện tại thì thoát vòng lặp
                break

    def fit_predict(self, X): #hàm phân cụm và dự đoán các điểm dữ liệu
        self.fit(X) #tìm ra K điểm tâm cụm
       
        return self._get_cluster_labels(self.clusters) #trả về danh sách nhãn các cụm của các điểm trong tập dữ liệu X
    def predict(self, X):
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters): #hàm gán nhãn cho các điểm theo tâm cụm được chia
      
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


    def _create_clusters(self, centroids): #tạo danh sách các mảng chứa các điểm có chung một cụm

        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids) #Lấy chỉ số của tâm cụm mà điểm đó gần nhất
            clusters[centroid_idx].append(idx) #điểm nào sẽ có nhãn của cụm gần nhất
        return clusters

    def _closest_centroid(self, sample, centroids): #trả về chỉ số tâm cụm mà điểm Sample gần nhất

        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx


    def _get_centroids(self, clusters): #hàm tính trung bình các điểm trong cùng một cụm để tạo thành các tâm cụm mới 
        centroids = np.zeros((self.K, self.n_features)) #mảng  0 0 0 0 
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids): #hàm kiểu tra sự khác nhau giữa các tâm mới và cũ 

        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0 #trả về true nếu ko có bất kì điểm nào khác

    
    def clustering(self,sample): #phân cụm cho mẫu đưa vào
        distances = [euclidean_distance(sample, point) for point in self._centroids()]
        closest_idx = np.argmin(distances)
        return closest_idx 
    def _centroids(self):#hàm lấy ra K tâm cụm
        return self.centroids 

#lấy dữ liệu     
rawdata=pd.read_csv("E:\HAVE_STUDIED\Hoc_May(A)\Project\Data\customer_data.csv")
data=np.array(rawdata[['Sex','Marital status','Age','Education','Income','Occupation','Settlement size']].values)

cls = [] #danh sách các mô hình ứng với số K 
for i in range(2,11): #vòng lặp tìm ra K phù hợp từ 2-10
    tmp = KMeans(K=i,max_iters=300)
    tmp.fit(data)
    cls.append(tmp) #thêm vào danh sách
#tính các độ đo ứng với từng mô hình phân cụm
Sil_score=[round(metrics.silhouette_score(data,i.predict(data)),3) for i in cls] 
DB_score=[round(metrics.davies_bouldin_score(data,i.predict(data)),3) for i in cls]

def btnCLS(): #Nút phân cụm
    gender=genderChoosen.current() #lấy dữ liệu
    if(maritalChoosen.current() == 0):
        marital=0
    else:
        marital=1
    age=spin_age.get()
    education=educationChoosen.current()
    income=spin_Income.get()
    if(OccupationChoosen.current() >= 2):
        occupation=2
    else:
        occupation=OccupationChoosen.current()
    Settlement=SettlementChoosen.current()
    if(income=='' or age ==''): #kiểu tra nếu chưa nhập đủ mà đã nhấn nút
        messagebox.showerror('Lỗi','Vui lòng nhập đầy đủ thông')
        
    data_duDoan=np.array([gender,marital,age,education,income,occupation,Settlement]).reshape(1,-1).astype('int')
    ketqua = [i.clustering(data_duDoan)+1 for i in cls]
    for i in range(9): #hiển thị các phân cụm 
        kq=Label(form,text=ketqua[i])
        kq.grid(column=i+1,row=7,padx=20)
        
def btnSil():
    messagebox.showinfo('Số cụm tốt nhất theo độ đo Silhouette',np.argmax(Sil_score)+2)
def btnDB():
    messagebox.showinfo('Số cụm tốt nhất theo độ đo Davies and Boundlin',np.argmin(DB_score)+2)
#Tạo form

form  = Tk()
form.title('Phân nhóm khách hàng')
form.geometry('1400x300')

label_gender = Label(form,text="Giới tính:")
label_gender.grid(row = 1, column = 0)
genderChoosen = ttk.Combobox(form, width = 10)
genderChoosen['values'] = (' Nam', ' Nữ')
genderChoosen['state']='readonly'
genderChoosen.grid(column = 0, row = 2,pady=20)
genderChoosen.current(0)
#
label_marital = Label(form,text="Tình trạng hôn nhân:")
label_marital.grid(row = 1, column = 1)
maritalChoosen = ttk.Combobox(form, width = 10)
maritalChoosen['values'] = (' Độc thân', 'Ly dị','Ly thân','Đã kết hôn','Góa bụa')
maritalChoosen['state']='readonly'
maritalChoosen.grid(column = 1, row = 2,pady=20)
maritalChoosen.current(0)
#Age
label_gender = Label(form,text="Tuổi:")
label_gender.grid(row = 1, column = 2)
spin_age = Spinbox(form, from_=18, to=100, width=5,)
spin_age.grid(column=2,row=2,pady=20)
#Education
label_education = Label(form,text="Trình độ học vấn:")
label_education.grid(row = 1, column = 3)
educationChoosen = ttk.Combobox(form, width = 10)
educationChoosen['values'] = (' Không rõ', 'THPT','Đang học Đại học','Tốt nghiệp Đại học')
educationChoosen['state']='readonly'
educationChoosen.grid(column = 3, row = 2,pady=20)
educationChoosen.current(0)
#Income
label_Income = Label(form,text="Thu nhập:")
label_Income.grid(row = 1, column = 4)
spin_Income = Spinbox(form, from_=0, to=1000000, width=5,)
spin_Income.grid(column=4,row=2,pady=20)
#Occupation
label_Occupation = Label(form,text="Nghề nghiệp:")
label_Occupation.grid(row = 1, column = 5)
OccupationChoosen = ttk.Combobox(form, width = 10)
OccupationChoosen['values'] = ('Thất nghiệp', 'Nhân viên văn phòng','Quản lý','Tự làm chủ','Nhân viên','Nhân viên có trình độ cao')
OccupationChoosen['state']='readonly'
OccupationChoosen.grid(column = 5, row = 2,pady=20)
OccupationChoosen.current(0)
#Settlement size
label_Settlement = Label(form,text="Nơi định cư:")
label_Settlement.grid(row = 1, column = 6)
SettlementChoosen = ttk.Combobox(form, width = 10)
SettlementChoosen['values'] = ('Thành phố nhỏ', 'Thành phố cỡ vừa','Thành phố lớn')
SettlementChoosen['state']='readonly'
SettlementChoosen.grid(column = 6, row = 2,pady=20)
SettlementChoosen.current(0)

button_cls=Button(form,text="Phân nhóm",background='blue',command=btnCLS)
button_cls.grid(row=3,column=2,padx=50)
#
lbl_k=Label(form,text='Số cụm:')
lbl_k.grid(column=0,row=4)
lbl_sil=Label(form,text='Silhousette score:')
lbl_sil.grid(column=0,row=5)
lbl_db=Label(form,text='David and Boundlin score:')
lbl_db.grid(column=0,row=6)
lbl_pred=Label(form,text='Thuộc cụm số:')
lbl_pred.grid(column=0,row=7)
#
for i in range(2,11):
    tmp=Label(form,text=i)
    tmp.grid(column=i-1,row=4,padx=20)
for i in range(9):
    sil=Label(form,text=Sil_score[i])
    sil.grid(column=i+1,row=5,padx=20)
    db=Label(form,text=DB_score[i])
    db.grid(column=i+1,row=6,padx=20)
#Tạo nút
button_findK_by_Sil=Button(form,text="Số cụm tốt nhất - Silhouette",background='blue',command=btnSil)
button_findK_by_Sil.grid(row=3,column=1,padx=50)
#
button_findK_by_DB=Button(form,text="Số cụm tốt nhất - DB",background='blue',command=btnDB)
button_findK_by_DB.grid(row=3,column=3,padx=50)

form.mainloop()

