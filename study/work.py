import sys
import os
import itertools
import mplcursors
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import rcParams
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5.QtWidgets import QMenu, QToolBar, QInputDialog, QAction, QProgressDialog, QSplitter, QTableWidget, QDialog, QFileDialog, QMessageBox,QVBoxLayout, QLabel, QListWidget
from PyQt5.QtCore import QCoreApplication
from scipy.stats import ttest_ind, f_oneway, pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage
from PyQt5.QtWidgets import QTableWidgetItem, QPushButton,QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import roc_curve, auc, RocCurveDisplay 
from sklearn.cross_decomposition import PLSRegression
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity  
# 配置字体支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体（黑体）
rcParams['axes.unicode_minus'] = False   # 避免负号显示错误

class MyThread(QThread):
    finished_signal = pyqtSignal(pd.DataFrame, str)  # 用于发送完成信号

    def __init__(self, data, method, parent=None):
        super(MyThread, self).__init__(parent)
        self.data = data
        self.method = method
        self.stop_thread = False  # 初始化停止标志

    def run(self):
        try:
            if self.stop_thread:
                return  # 停止线程
            result = pd.DataFrame()  # 将 result 初始化为一个空的 DataFrame
            if self.method == "Mean Fill":
                result = self.mean_impute(self.data)
            elif self.method == "Median Fill":
                result = self.median_impute(self.data)
            elif self.method == "Low Fill":
                result = self.low_value_impute(self.data)
            elif self.method == "RF Fill":
                result = self.random_forest_impute(self.data)
            elif self.method == "KNN Fill":
                result = self.knn_impute(self.data)
            elif self.method == "BR Fill":
                result = self.bayesian_ridge_impute(self.data)
            elif self.method == "SVR Fill":
                result = self.svr_impute(self.data)
            else:
                result = self.data.copy()  # 若无有效方法，返回原数据
            
            if result.empty:
                result = pd.DataFrame()

            self.finished_signal.emit(result, self.method)
        except Exception as e:
            self.finished_signal.emit(None, str(e))

    def stop(self):
        """停止线程"""
        self.stop_thread = True

    def random_forest_impute(self, data):
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    def knn_impute(self, data):
        imputer = KNNImputer(n_neighbors=5)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def bayesian_ridge_impute(self, data):
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    def svr_impute(self, data):
        imputer = IterativeImputer(estimator=SVR(), max_iter=10, random_state=0)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    def low_value_impute(self, data):
        return data.fillna(0)

    def mean_impute(self, data):
        return data.fillna(data.mean())

    def median_impute(self, data):
        return data.fillna(data.median())

class FirstWindow(QtWidgets.QMainWindow):
    menu_toggle_signal = pyqtSignal(bool)
    def __init__(self):
        super().__init__()
        uic.loadUi('First_window.ui', self)  # 使用正确的路径
        self.editable_data = None
        self.original_data = None
        self.stop_normalization = False  # 初始化终止标志
        self.toolBar = Toolbar(self,self.original_data, self.toolBar1, self.toolBar2, self.toolBar3, 
                               self.action_one_factor, self.action_multifactorial, 
                               self.action_biomarker_analysis, self.actionFold_Change, self.actionT_test,
                               self.actionVolcano_Plot, self.actionANOVA, self.actionCorrelations, self.actionPCA, 
                               self.actionPLS, self.actionFactor_Analysis, self.actionDiscriminant_Analysis, 
                               self.actionCluster_Analysis, self.tableWidget, self.tableWidget_2, self.editable_data,
                               self.action_analysis, self.action_process, self.stackedWidget)
        
        
        # 查找两个 QTableWidget（确保是 QTableWidget）
        table_widget1 = self.findChild(QTableWidget, 'tableWidget')  
        table_widget2 = self.findChild(QTableWidget, 'tableWidget_2')  

        # 如果查找到的 table_widget1 和 table_widget2 是有效的
        if table_widget1 and table_widget2: 
            # 创建 QSplitter 并将两个 QTableWidget 添加到其中
            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(table_widget1)
            splitter.addWidget(table_widget2)

            
            Widget = self.findChild(QWidget, 'widget')  
            layout = Widget.layout()  # 获取现有布局，不新建，避免冲突

            # 确保布局存在并将 QSplitter 添加到布局中
            if layout:
                layout.addWidget(splitter)
                Widget.setLayout(layout)

            # 将 QSplitter 添加到布局中
            Widget.layout().addWidget(splitter)

        # 初始化窗口
        self.radioButton_Low.setChecked(True)
        self.radioNoneButton.setChecked(True)
        self.stackedWidget.setCurrentIndex(0)
        
        self.file_path = None
        self.data = None

        self.filePathLineEdit.setPlaceholderText("Please upload a file...")
        self.pushButton_refresh.clicked.connect(self.refresh_table_data)
        self.pushButton_save.clicked.connect(self.save_table_data)
        self.pushButton_EditGroup.clicked.connect(self.Group_table_data)
        self.pushButton_Proceed.clicked.connect(self.proceed_cb)
        self.action_read.triggered.connect(self.open_file)

        # 查找名为 'menu' 的菜单项
        self.menu= self.menubar.findChild(QtWidgets.QMenu, 'menuFile')
        self.actionOpen = self.menubar.findChild(QtWidgets.QMenu, 'actionopen')

        if self.menu:
            self.pushButton_upload.clicked.connect(self.menu_item_triggered)
        else:
            print("pushButton_upload not found!")

        # 在初始化函数中绑定事件
        menu_helps = self.findChild(QAction, "action_read")  # 查找 read 对象
        if menu_helps:  

            menu_helps.triggered.connect(self.open_manual)
        else:
                QMessageBox.warning(self, "Warning", "Action item 'action_read' not found.")
    
    def menu_item_triggered(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV or TXT files (*.csv *.txt)")
        self.editable_data = None
        
        if not self.file_path:
            QMessageBox.critical(self, "Error", "Please upload a file")
            return
        else:
            self.filePathLineEdit.setText(self.file_path)



        if self.file_path:
            try:
                if self.file_path.endswith('.csv'):
                    data = pd.read_csv(self.file_path)
                elif self.file_path.endswith('.txt'):
                    data = pd.read_csv(self.file_path, delimiter='\t')
                else:
                    QMessageBox.critical(self, "Error", "Unsupported file type")
                    return
                
                # 统计缺失值

                missing_values_count = data.isnull().sum().sum()
                QMessageBox.information(self, "File Information", f"File loaded successfully!\nTotal missing values: {missing_values_count}")
                
                self.original_data = data.copy()
                self.labels = data.iloc[:, 1]
                self.data = data.drop(columns=[data.columns[1]])
                self.data = self.data.select_dtypes(include=[np.number])
                
                # 更新 Toolbar 数据
                self.toolBar.update_data(self.original_data)
                if self.data.empty:
                    QMessageBox.critical(self, "Error", "No numeric columns found in the file")
                    return

                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the file: {str(e)}")
            
            
        if self.file_path:
            try:
                
                selected_method = self.get_selected_method()
                selected_normalization = self.get_selected_normalization()
                
                
                self.process_file(self.file_path, selected_method, selected_normalization)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred during file processing: {str(e)}")
    
            #跳转到下一个界面
            self.stackedWidget.setCurrentIndex(1)

            for menu in self.menuBar().children():
                if isinstance(menu, QMenu):  # 确保是 QMenu 对象
                    # 遍历菜单中的所有 Action
                    for action in menu.actions():
                        action.setEnabled(False)
                        
            for toolbar in self.children():  # 遍历窗口的所有子控件
                if isinstance(toolbar, QToolBar):  # 确保是 QToolBar
                    for action in toolbar.actions():  # 遍历工具栏中的所有 QAction
                        action.setEnabled(False)  # 禁用 QAction


    def open_file(self):
        current_dir = os.getcwd()
        self.file_path11 = os.path.join(current_dir, "read.doc")
        
    def refresh_table_data(self):
        """点击刷新按钮后更新表格数据"""
        if self.data is None:
            QMessageBox.critical(self, "Error", "Please upload a file first")
            return

        self.stop_normalization = False  # 初始化终止标志
        self.lock_buttons(True)  # 锁定界面上的按钮

        total_steps = 1000  # 假设有1000个步骤print

        # 创建进度对话框
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setWindowTitle("Please wait")
        self.progress_dialog.setLabelText("Processing data...")
        self.progress_dialog.setRange(0, total_steps)
        self.progress_dialog.setWindowFlags(self.progress_dialog.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)

        # 添加自定义取消按钮
        cancel_button = QPushButton("Cancel", self.progress_dialog)
        cancel_button.clicked.connect(self.on_cancel_clicked)  # 明确绑定
        self.progress_dialog.setCancelButton(cancel_button)
        self.progress_dialog.show()

        try:
            selected_method = self.get_selected_method()
            selected_normalization = self.get_selected_normalization()

            # 启动填充线程
            self.thread = MyThread(self.data, selected_method)
            self.thread.finished_signal.connect(lambda filled_data, method: self.on_fill_method_done(filled_data, method, selected_normalization))
            self.thread.start()

            # 模拟处理数据的过程
            for step in range(total_steps):
                
                if self.stop_normalization:
                    
                    break

                # 调用数据处理函数
                self.process_step(step, selected_method, selected_normalization)

                # 更新进度条
                self.progress_dialog.setValue(step + 1)

                # 保持界面响应
                QCoreApplication.processEvents()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during data processing: {str(e)}")

        finally:
            # 进入finally，解锁界面上的其它按钮
            self.lock_buttons(False)

            # 确保进度对话框关闭
            if hasattr(self, 'progress_dialog') and self.progress_dialog.isVisible():
                self.progress_dialog.close()

    def on_cancel_clicked(self):
        """处理用户点击取消按钮的逻辑"""
        self.stop_normalization = True  # 设置停止标志
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()  # 停止填充线程
        self.progress_dialog.close()  # 确保关闭进度对话框
        QMessageBox.critical(self, "Processing Terminated by User", "The process has been canceled by the user")

    def lock_buttons(self, lock):
        """锁定或解锁界面上的按钮"""
        self.pushButton_refresh.setDisabled(lock)
        self.pushButton_EditGroup.setDisabled(lock)
        self.pushButton_Proceed.setDisabled(lock)
        
    def process_step(self, step, selected_method, selected_normalization):
        """模拟每一步的数据处理过程"""
        import time
        time.sleep(0.05)  # 模拟每一步耗时操作

    def get_selected_method(self):
        
        if self.radioButton_Low.isChecked():
            return "Low Fill"
        elif self.radioButton_Mean.isChecked():
            return "Mean Fill"
        elif self.radioButton_Median.isChecked():
            return "Median Fill"
        elif self.radioButton_KNN.isChecked():
            return "KNN Fill"
        elif self.radioButton_RF.isChecked():
            return "RF Fill"
        elif self.radioButton_BR.isChecked():
            return "BR Fill"
        elif self.radioButton_SVR.isChecked():
            return "SVR Fill"
        else:
            return "No Method Selected"

    def get_selected_normalization(self):
        
        if self.radioNoneButton.isChecked():
            return "None"
        elif self.radioMinMaxButton.isChecked():
            return "Min-Max"
        elif self.radioZScoreButton.isChecked():
            return "Z-Score"
        elif self.radioMeanButton.isChecked():
            return "Mean Norm"
        elif self.radioMedianButton.isChecked():
            return "Median Norm"
        else:
            return "No Normalization Selected"

    def process_file(self, file_path, method, normalization):
    # 读取文件
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.txt'):
                data = pd.read_csv(file_path, delimiter='\t')
            else:
                QMessageBox.critical(self, "Error", "Unsupported file type")
                return
            
        
            # 提取标签和数据
            if data.shape[1] > 1:  # 确保数据有足够的列
                
                data = data.drop(columns=[data.columns[1]])
                data = data.select_dtypes(include=[np.number])  # 只选择数值列
            else:
                QMessageBox.critical(self, "Error", "The file does not contain enough columns")
                return
            
             # 启动线程来填充缺失值
            self.thread = MyThread(data, method)
            self.thread.finished_signal.connect(lambda filled_data, method: self.on_fill_method_done(filled_data, method, normalization))
            self.thread.start()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during file processing: {str(e)}")

    def on_fill_method_done(self, filled_data, method, normalization):
        
        if self.stop_normalization:
            return  # 如果终止标志已设置，直接退出
    
        if filled_data is not None:
            
            # 完成缺失值填充后执行归一化
            if normalization == "Min-Max":
                filled_data = (filled_data - filled_data.min()) / (filled_data.max() - filled_data.min())
            elif normalization == "Z-Score":
                filled_data = (filled_data - filled_data.mean()) / filled_data.std()
            elif normalization == "Mean Norm":
                filled_data = (filled_data - filled_data.mean()) / (filled_data.max() - filled_data.min())
            elif normalization == "Median Norm":
                filled_data = (filled_data - filled_data.median()) / (filled_data.max() - filled_data.min())
            elif normalization == "None":
                pass  # Do nothing if "None" is selected

            # 在界面中的 QTableWidget 显示结果
            
            self.tableWidget.clear()
            self.tableWidget.setRowCount(filled_data.shape[0])
            self.tableWidget.setColumnCount(filled_data.shape[1])
            self.tableWidget.setHorizontalHeaderLabels(filled_data.columns)

            for row in range(filled_data.shape[0]):
                for col in range(filled_data.shape[1]):
                    formatted_value = "{:.4g}".format(filled_data.iat[row, col])
                    item = QtWidgets.QTableWidgetItem(formatted_value)
                    self.tableWidget.setItem(row, col, item) 

                self.tableWidget.resizeRowToContents(row)

            for col in range(filled_data.shape[1]):
                self.tableWidget.resizeColumnToContents(col)
                

            #获取第三个窗口中的 tableWidget

            self.tableWidget_3.setRowCount(filled_data.shape[0])
            self.tableWidget_3.setColumnCount(filled_data.shape[1])
            self.tableWidget_3.setHorizontalHeaderLabels(filled_data.columns)

            # 填充表格数据
            for i in range(filled_data.shape[0]):  # 行 
                for j in range(filled_data.shape[1]):  # 列
                    # 格式化为四位有效数字
                    formatted_value = "{:.4g}".format(filled_data.iat[i, j])
                    self.tableWidget_3.setItem(i, j, QTableWidgetItem(formatted_value))
                self.tableWidget_3.resizeRowToContents(i)

            for col in range(filled_data.shape[1]):
                self.tableWidget_3.resizeColumnToContents(col)
            

            QMessageBox.information(self, "Success", f"Data processed and displayed successfully (Filling Method: {method}, Normalization Method: {normalization})")
        
            # 确保进度对话框关闭
            if hasattr(self, 'progress_dialog') and self.progress_dialog.isVisible():
                self.progress_dialog.close()
                 # 更新 stop_normalization 标志为 True
                self.stop_normalization = True  # 显式设置为True，确保后续逻辑能正确执行
                

           
        else:
            QMessageBox.critical(self, "Error", "Data processing failed")
        
        self.stored_filled_data = filled_data  # 存储传递的参数

    def save_table_data(self):
        # 弹出选项对话框询问用户保存类型
        options = ["1st window data", "2nd window data", "All data"]
        choice, ok = QInputDialog.getItem(self, "Save Option", "Select data to save:", options, 0, False)
        if not ok:
            return
        
        # 弹出文件保存对话框，让用户选择保存路径
        def save_single_table(table, window_name):
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                f"保存{window_name}", 
                "", 
                "CSV files (*.csv);;Text files (*.txt)"
            )
            if not file_path:
                return False

            # 获取表格数据
            row_count = table.rowCount()
            column_count = table.columnCount()
            data = []
            for row in range(row_count):
                row_data = []
                for col in range(column_count):
                    item = table.item(row, col)
                    row_data.append(item.text() if item else "")
                data.append(row_data)

            # 创建DataFrame并保存
            try:
                df = pd.DataFrame(
                    data,
                    columns=[table.horizontalHeaderItem(col).text() for col in range(column_count)]
                )
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                elif file_path.endswith('.txt'):
                    df.to_csv(file_path, sep='\t', index=False)
                QMessageBox.information(self, "成功", f"{window_name}保存成功！")
                return True
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                return False

        if choice == "1st window data":
            save_single_table(self.tableWidget, "1st window data")
        elif choice == "2nd window data":
            save_single_table(self.tableWidget_2, "2nd window data")
        elif choice == "All data":
            save_single_table(self.tableWidget, "1st window data")
            save_single_table(self.tableWidget_2, "2nd window data")
    
    def Group_table_data(self):
        if self.editable_data is None:
            self.editable_data = self.original_data.copy()
        self.group = None

        if self.labels is None:
            QMessageBox.warning(self, "Warning", "No group data available")
            return
        
        
        # 创建弹窗
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Groups")
        dialog.setModal(True)

        dialog.resize(400, 600)

        # 布局
        layout = QVBoxLayout(dialog)

        # 显示当前的组别数据
        label = QLabel("Please edit the groups")
        layout.addWidget(label)

        # 创建 QListWidget 用于显示和编辑第一列数据
        tableWidget = QTableWidget(dialog)
        tableWidget.setRowCount(len(self.labels))  # 设置行数
        tableWidget.setColumnCount(2)  # 设置列数（显示两列）
        tableWidget.setHorizontalHeaderLabels(["Samples", "Label"])  # 设置表头
        
        
        # 填充数据到 QTableWidget 中
        for row in range(len(self.editable_data)):
            # 第一列显示组别数据
            group_item = QTableWidgetItem(str(self.editable_data.iloc[row,0]))
            tableWidget.setItem(row, 0, group_item)
            
            # 第二列显示数据（可以根据需要填充 self.data 的对应列）
            data_item = QTableWidgetItem(str(self.editable_data.iloc[row,1]))  # 显示 self.data 的第一列
            tableWidget.setItem(row, 1, data_item)

        layout.addWidget(tableWidget)
        
        # 保存按钮
        saveButton = QPushButton("Save", dialog)
        layout.addWidget(saveButton)
        
        # 绑定保存按钮的功能
        def save_groups():
            self.group = True
            # 获取用户修改后的组别
            modified_groups = [tableWidget.item(i, 1).text() for i in range(tableWidget.rowCount())]
            
            # 检查是否有缺失值并恢复缺失的组别
            has_empty = False
            for row in range(len(modified_groups)):
                if modified_groups[row] == "":
                    has_empty = True
                    # 恢复原有的组别信息
                    modified_groups[row] = self.editable_data.iloc[row, 1]  # 恢复原有组别
                    tableWidget.item(row, 1).setText(modified_groups[row])  # 更新显示的组别
                else:
                    # 如果组别不为空，确保将其转换为字符串
                    tableWidget.item(row, 1).setText(str(modified_groups[row]))

            if has_empty:
                QMessageBox.warning(self, "Warning", "Group information cannot be empty. Missing groups have been restored to their original data.")
                return  # 如果有缺失值，退出函数，不进行更新

            # 将修改后的分组信息更新到原数据表中
            for row in range(len(modified_groups)):
                self.editable_data.iloc[row, 1] = modified_groups[row]
            
            # 统计组别信息
            group_counts = Counter(modified_groups)
            group_summary = "\n".join([f"{group}: {count} " for group, count in group_counts.items()])

            # 显示成功消息和分组信息
            QMessageBox.information(self, "Success", f"Group assignment has been updated!\n\nGroups are as follows:\n{group_summary}")

            dialog.accept()

            # **调用 display_combined_data**

            if hasattr(self, "stored_filled_data"):
                self.sample_data = pd.DataFrame(self.original_data.iloc[:, 0])
                self.modified_groups = pd.DataFrame(modified_groups, columns=['Label'])
                self.display_combined_data(self.sample_data, self.stored_filled_data, self.modified_groups)               

            
        saveButton.clicked.connect(save_groups)

        self.toolBar.editable_data = self.editable_data
        
        dialog.exec_()

    def proceed_cb(self):
        if self.editable_data is None or self.group is None:
            QMessageBox.critical(self, "Error", "Please edit the group information") 
            return
        else:   
            self.tableWidget_2.clearContents()
            self.tableWidget_2.setRowCount(0)
            self.tableWidget_2.setColumnCount(0)
            
            self.stackedWidget.setCurrentIndex(0)

            for menu in self.menuBar().children():
                if isinstance(menu, QMenu):  # 确保是 QMenu 对象
                    # 遍历菜单中的所有 Action
                    for action in menu.actions():
                        action.setEnabled(True)

            for toolbar in self.children():  # 遍历窗口的所有子控件
                if isinstance(toolbar, QToolBar):  # 确保是 QToolBar
                    for action in toolbar.actions():  # 遍历工具栏中的所有 QAction
                        action.setEnabled(True)

            self.display_combined_data(self.sample_data, self.stored_filled_data, self.modified_groups)

    def display_combined_data(self, sample_data, filled_data, modified_groups):
        """
        将 sample_data、filled_data 和 modified_groups 拼接后显示在 tableWidget 中
        """
        self.tableWidget.clear()  # 清空现有表格数据

        try:
            # 确保传入参数为 Pandas DataFrame
            if not all(isinstance(arg, pd.DataFrame) for arg in [sample_data, filled_data, modified_groups]):
                QMessageBox.critical(self, "Error", "The provided data is not a valid Pandas DataFrame")
                return
            
            # 确保行数一致
            if not (sample_data.shape[0] == filled_data.shape[0] == modified_groups.shape[0]):
                QMessageBox.critical(self, "Error", "Row counts are inconsistent; data cannot be concatenated")

                return
            
            # 拼接数据
            combined_data = pd.concat([sample_data, modified_groups, filled_data], axis=1)

            # 检查拼接后的数据与 self.original_data 的维度是否一致
            original_shape = self.original_data.shape
            combined_shape = combined_data.shape
            
            if original_shape != combined_shape:
                 # 获取缺失的列
                missing_columns = set(self.original_data.columns) - set(combined_data.columns)
                if missing_columns:
                    # 生成缺失列的信息
                    missing_info = "\n".join([f"Missing column: '{col}' at index {self.original_data.columns.get_loc(col)}"
                                            for col in missing_columns])
                    # 显示警告信息
                    QMessageBox.warning(
                        None,  # 这里传入窗口对象
                        "Warning",
                        f"Data mismatch detected!\n"
                        f"Original data shape: {original_shape}\n"
                        f"Combined data shape: {combined_shape}\n"
                        f"Missing columns:\n{missing_info}\n"
                        f"Please check your data."
                    )
                return  # 终止后续操作
            
            # 更新 tableWidget 的行数和列数
            self.tableWidget.setRowCount(combined_data.shape[0])
            self.tableWidget.setColumnCount(combined_data.shape[1])
            
            # 设置表头
            self.tableWidget.setHorizontalHeaderLabels(combined_data.columns)
            
            # 填充表格数据
            for row in range(combined_data.shape[0]):
                for col in range(combined_data.shape[1]):
                    # 获取单元格数据并格式化
                    cell_value = combined_data.iat[row, col]
                    formatted_value = (
                        "{:.4g}".format(cell_value) 
                        if pd.api.types.is_numeric_dtype(type(cell_value)) 
                        else str(cell_value)
                    )
                    self.tableWidget.setItem(row, col, QTableWidgetItem(formatted_value))
                
                # 调整行高度
                self.tableWidget.resizeRowToContents(row)
            
            # 调整列宽度
            for col in range(combined_data.shape[1]):
                self.tableWidget.resizeColumnToContents(col)
            
            QMessageBox.information(self, "Success", "Data has been successfully concatenated and displayed on the main page")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while displaying the concatenated data: {str(e)}")
 
    def open_manual(self):
    # 定义说明书文件的路径
        manual_path = os.path.join(os.getcwd(), "read.doc")
        if not os.path.exists(manual_path):
            QMessageBox.critical(self, "Error", f"The manual file does not exist:\n{manual_path}")
            return
        
        try:
            # 打开文件（Windows 平台）
            os.startfile(manual_path)
        except AttributeError:
            # 对于其他平台，使用 subprocess 打开
            import subprocess
            subprocess.Popen(["open", manual_path] if sys.platform == "darwin" else ["xdg-open", manual_path])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open manual:\n{str(e)}")
  
class Toolbar:
    def __init__(self, parent, original_data,toolbar1, toolbar2, toolbar3, action_one_factor, 
                 action_multifactorial, action_biomarker_analysis, actionFold_Change, actionT_test,
                 actionVolcano_Plot,actionANOVA,actionCorrelations,actionPCA, actionPLS, actionFactor_Analysis, 
                 actionDiscriminant_Analysis, actionCluster_Analysis, tableWidget, tableWidget_2 ,editable_data,
                 action_analysis, action_process, stackedWidget):
        self.parent = parent
        self.editable_data = editable_data
        self.toolBar1 = toolbar1
        self.toolBar2 = toolbar2
        self.toolBar3 = toolbar3
        self.action_one_factor = action_one_factor
        self.action_multifactorial = action_multifactorial
        self.action_biomarker_analysis = action_biomarker_analysis
        self.actionFold_Change = actionFold_Change
        self.actionT_test = actionT_test
        self.actionVolcano_Plot = actionVolcano_Plot
        self.actionANOVA = actionANOVA
        self.actionCorrelations = actionCorrelations
        self.actionPCA = actionPCA
        self.actionPLS = actionPLS
        self.actionFactor_Analysis = actionFactor_Analysis
        self.actionDiscriminant_Analysis = actionDiscriminant_Analysis
        self.actionCluster_Analysis = actionCluster_Analysis
        self.tableWidget = tableWidget
        self.tableWidget_2 = tableWidget_2
        self.original_data = original_data
        self.action_analysis = action_analysis
        self.action_process = action_process
        self.stackedWidget = stackedWidget


        self.toolBar1.hide()
        self.toolBar2.hide()
        self.toolBar3.hide()

        # 连接 QAction 到切换工具栏的槽函数
        self.action_one_factor.triggered.connect(self.showToolBar1)
        self.action_multifactorial.triggered.connect(self.showToolBar2)
        self.action_biomarker_analysis.triggered.connect(self.showToolBar3)
        self.actionFold_Change.triggered.connect(self.Fold_Change_cb)
        self.actionT_test.triggered.connect(self.T_test_cb)
        self.actionVolcano_Plot.triggered.connect(self.Volcano_Plot_cb)
        self.actionANOVA.triggered.connect(self.ANOVA_cb)
        self.actionCorrelations.triggered.connect(self.Correlations_cb)
        self.actionPCA.triggered.connect(self.PCA_cb)
        self.actionPLS.triggered.connect(self.PLS_cb)
        self.actionFactor_Analysis.triggered.connect(self.Factor_Analysis_cb)
        self.actionDiscriminant_Analysis.triggered.connect(self.Discriminant_Analysis_cb)
        self.actionCluster_Analysis.triggered.connect(self.Cluster_Analysis_cb)
        self.action_analysis.triggered.connect(self.page1)
        self.action_process.triggered.connect(self.page2)
        
        # 设置样式表
        self.toolBar1.setStyleSheet("QToolButton { font-size: 24px; font-family: Times New Roman; }")
        self.toolBar2.setStyleSheet("QToolButton { font-size: 24px; font-family: Times New Roman; }")
        self.toolBar3.setStyleSheet("QToolButton { font-size: 24px; font-family: Times New Roman; }")

        self.tableWidget_2.clear()

    def update_data(self, new_data):
        self.original_data = new_data
        
    def page1(self):
        self.stackedWidget.setCurrentIndex(0)

    def page2(self):
        self.stackedWidget.setCurrentIndex(1)

    def showToolBar1(self):
        self.toolBar1.show()
        self.toolBar2.hide()
        self.toolBar3.hide()
        
    def showToolBar2(self):
        self.toolBar1.hide()
        self.toolBar2.show()
        self.toolBar3.hide()

    def showToolBar3(self):
        self.toolBar1.hide()
        self.toolBar2.hide()
        self.toolBar3.show()

    def Fold_Change_cb(self):

        # 从 tableWidget 中提取数据
        group1 = []
        group2 = []
        # 提取物质名称，从第三列开始
        identifiers = list(self.editable_data.columns[2:])
            
        groups = [self.editable_data.iloc[i, 1] for i in range(self.editable_data.shape[0])]
        
        
        numCols = self.tableWidget.columnCount() 

        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to calculate Fold Change")
            return

        try:
           
            # 创建组别的字典
            group_indices = {}
            for index, group in enumerate(groups):
                if group not in group_indices:
                    group_indices[group] = []
                group_indices[group].append(index)

            # 确保有两个不同的组别
            if len(group_indices) != 2:
                QMessageBox.warning(self.tableWidget, "Error", "Fold Change analysis requires exactly two distinct groups")
                return

            # 获取不同的组别
            unique_groups = list(group_indices.keys())


            # 定义fold change结果存储
            fold_change_results_all = {}
            
            # 两两比较各组之间的fold change
            for (group1, group2) in itertools.combinations(unique_groups, 2):
                group1_indices = group_indices[group1]
                group2_indices = group_indices[group2]
                
                all_mean1, all_std1, all_mean2, all_std2, fold_change_results = [], [], [], [], []
                
                for col in range(2, numCols):  # 从第三列开始
                    group1_data = [float(self.tableWidget.item(row, col).text()) for row in group1_indices]
                    group2_data = [float(self.tableWidget.item(row, col).text()) for row in group2_indices]

                    mean1, std1 = self.calculate_mean_std(group1_data)
                    mean2, std2 = self.calculate_mean_std(group2_data)
                    
                    all_mean1.append(mean1)
                    all_std1.append(std1)
                    all_mean2.append(mean2)
                    all_std2.append(std2)
                    
                    
                    fold_change_result = self.calculate_fold_change(all_mean1, all_mean2)
                    # print(type(fold_change_result))
                    fold_change_results.append(fold_change_result)
                
                # 保存结果到字典
                fold_change_results_all[(group1, group2)] = (identifiers, all_mean1, all_std1, all_mean2, fold_change_results)

            # 将结果展示到tableWidget_2中
            self.display_fold_change_result(identifiers, all_mean1, all_std1, all_mean2, fold_change_results)
            

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def calculate_fold_change(self, all_mean1, all_mean2):
        fold_change_result = []
        for g1, g2 in zip(all_mean1, all_mean2):
            if g1 == 0:
                fold_change_result.append(0)
            else:
                fold_change_result.append(g2 / g1)  # 计算 fold_change▎
        return fold_change_result

    def calculate_mean_std(self, group):
        mean = sum(group) / len(group)
        std = (sum((x - mean) ** 2 for x in group) / (len(group) - 1)) ** 0.5
        return mean, std       

    def display_fold_change_result(self, identifiers, all_mean1, all_std1, all_mean2, fold_change_results):
        # 清空 tableWidget2
        self.tableWidget_2.clear()
        # print(fold_change_results)
        
        # 设置 tableWidget2 的行数和列数
        self.tableWidget_2.setRowCount(len(identifiers))
        self.tableWidget_2.setColumnCount(5)
        self.tableWidget_2.setHorizontalHeaderLabels(["Substance", "Mean (Group 1)", "Std Dev (Group 1)", "Mean (Group 2)", "Fold Change"])

        # 将每种物质的统计结果插入到 tableWidget2
        for row, (identifier, mean1, std1, mean2, fold_changes) in enumerate(
            zip(identifiers, all_mean1, all_std1, all_mean2, fold_change_results)
        ):
            # 添加物质名称到第一列
            item_id = QTableWidgetItem(str(identifier))
            self.tableWidget_2.setItem(row, 0, item_id)
            # print(identifiers)
            # 组1平均数
            item_mean1 = QTableWidgetItem(f"{mean1:.2f}")
            self.tableWidget_2.setItem(row, 1, item_mean1)

            # 组1标准差
            item_std1 = QTableWidgetItem(f"{std1:.2f}")
            self.tableWidget_2.setItem(row, 2, item_std1)

            # 组2平均数
            item_mean2 = QTableWidgetItem(f"{mean2:.2f}")
            self.tableWidget_2.setItem(row, 3, item_mean2)

            # Fold Change 结果
            
            
            fold_change_mean = fold_changes[-1] if fold_changes else 0
            item_fc = QTableWidgetItem(f"{fold_change_mean:.2f}")
            self.tableWidget_2.setItem(row, 4, item_fc)
 
            # 调整每行的高度以匹配该行中内容的高度
            self.tableWidget_2.resizeRowToContents(row) 

        # 调整列宽适合内容
        for col in range(5):
            self.tableWidget_2.resizeColumnToContents(col)

        QMessageBox.information(self.tableWidget_2, "Success", "Fold Change results have been updated")
     
    def T_test_cb(self):
        # 从 tableWidget 中提取数据
        identifiers = list(self.editable_data.columns[2:])
        
        groups = [self.editable_data.iloc[i, 1] for i in range(self.editable_data.shape[0])]
        
        numCols = self.tableWidget.columnCount() 

        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform t-test analysis")
            return

        try:
            # 创建组别的字典
            group_indices = {}
            for index, group in enumerate(groups):
                if group not in group_indices:
                    group_indices[group] = []
                group_indices[group].append(index)

            # 确保有两个不同的组别
            if len(group_indices) != 2:
                QMessageBox.warning(self.tableWidget, "Error", "t-test analysis requires exactly two distinct groups")
                return

            # 获取不同的组别
            unique_groups = list(group_indices.keys())

            # 定义t-test结果存储
            t_test_results_all = {}
            
            # 两两比较各组之间的t-test
            for (group1, group2) in itertools.combinations(unique_groups, 2):
                group1_indices = group_indices[group1]
                group2_indices = group_indices[group2]
                
                t_values, p_values = [], []
                
                for col in range(2, numCols):  
                    group1_data = [float(self.tableWidget.item(row, col).text()) for row in group1_indices]
                    group2_data = [float(self.tableWidget.item(row, col).text()) for row in group2_indices]

                    # 计算t-test和p-value
                    t_stat, p_value = self.perform_t_test(group1_data, group2_data)
                    
                    t_values.append(t_stat)
                    p_values.append(p_value)

                # 保存结果到字典
                t_test_results_all[(group1, group2)] = (identifiers, t_values, p_values)

            # 将结果展示到tableWidget_2中
            self.display_t_test_result(identifiers, t_values, p_values)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def perform_t_test(self, group1, group2):
        # 计算 t-test
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        return t_stat, p_value

    def display_t_test_result(self, identifiers, t_values, p_values):
        # 清空 tableWidget2
        self.tableWidget_2.clear()

        # 设置 tableWidget2 的行数和列数
        self.tableWidget_2.setRowCount(len(identifiers))
        self.tableWidget_2.setColumnCount(3)
        self.tableWidget_2.setHorizontalHeaderLabels(["Substance", "T Value", "P Value"])

        # 填充 t 和 p 值到 tableWidget2
        for row, (identifier, t_value, p_value) in enumerate(
            zip(identifiers, t_values, p_values)):
            # 物质名称
            item_id = QTableWidgetItem(str(identifier))
            self.tableWidget_2.setItem(row, 0, item_id)

            # T 值
            item_t = QTableWidgetItem(f"{t_value:.2f}")
            self.tableWidget_2.setItem(row, 1, item_t)

            # P 值
            item_p = QTableWidgetItem(f"{p_value:.4f}")
            self.tableWidget_2.setItem(row, 2, item_p)

            # 调整每行的高度以适应内容
            self.tableWidget_2.resizeRowToContents(row)

        # 调整列宽适应内容
        for col in range(3):
            self.tableWidget_2.resizeColumnToContents(col)

        QMessageBox.information(self.tableWidget_2, "Success", "T-test results have been updated")

    def Volcano_Plot_cb(self):
        identifiers = list(self.editable_data.columns[2:])  # 从第三列开始提取物质名称
        groups = [self.editable_data.iloc[i, 1] for i in range(self.editable_data.shape[0])]
        
        numCols = self.tableWidget.columnCount()
        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform Volcano Plot analysis")
            return

        fold_changes = []
        p_values = []

        try:
            # 创建组别的字典
            group_indices = {}
            for index, group in enumerate(groups):
                if group not in group_indices:
                    group_indices[group] = []
                group_indices[group].append(index)

            # 确保有两个不同的组别
            if len(group_indices) != 2:
                QMessageBox.warning(self.tableWidget, "Error", "Volcano Plot analysis requires exactly two distinct groups")
                return

            # 获取不同的组别
            unique_groups = list(group_indices.keys())
            group1_indices = group_indices[unique_groups[0]]
            group2_indices = group_indices[unique_groups[1]]

            # 逐列计算 Fold Change 和 p 值
            for col in range(2, numCols):  # 从第三列开始
                group1_data = [float(self.tableWidget.item(row, col).text()) for row in group1_indices]
                group2_data = [float(self.tableWidget.item(row, col).text()) for row in group2_indices]

                fold_change = self.calculate_fold_change(group1_data, group2_data)
                fold_change_mean = sum(fold_change) / len(fold_change) if fold_change else 0
                fold_changes.append(fold_change_mean)

                t_stat, p_value = self.perform_t_test(group1_data, group2_data)
                p_values.append(p_value)

            # 展示火山图
            self.show_volcano_plot_dialog(identifiers, fold_changes, p_values)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def show_volcano_plot_dialog(self, identifiers, fold_changes, p_values):
        # 创建对数数据
        log2_fold_changes = np.log2(fold_changes)
        neg_log10_p_values = -np.log10(p_values)

        # 创建弹窗
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Volcano Plot")
        layout = QVBoxLayout(dialog)

        # 创建 Matplotlib 图形
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(log2_fold_changes, neg_log10_p_values, color='gray', alpha=0.7)

        # 标记显著性
        significant = (neg_log10_p_values > 1.3) & (abs(log2_fold_changes) > 1)
        ax.scatter(np.array(log2_fold_changes)[significant], np.array(neg_log10_p_values)[significant], color='red', alpha=0.7, label='Significant')

        # 设置轴标签和标题
        ax.set_xlabel('Log2(Fold Change)')
        ax.set_ylabel('-Log10(p-value)')
        ax.set_title('Volcano Plot')
        ax.axhline(y=1.3, color='blue', linestyle='--')
        ax.axvline(x=1, color='blue', linestyle='--')
        ax.axvline(x=-1, color='blue', linestyle='--')
        ax.legend()

        # 鼠标悬停效果
        annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
        annotation.set_visible(False)

        def on_hover(event):
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    annotation.set_visible(True)
                    index = ind["ind"][0]
                    x, y = log2_fold_changes[index], neg_log10_p_values[index]
                    # 显示物质名称和坐标信息
                    annotation.xy = (x, y)
                    annotation.set_text(f"{identifiers[index]}\n(Log2FC: {x:.2f}, -Log10P: {y:.2f})")
                    fig.canvas.draw_idle()
                else:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        # 将图表嵌入弹窗中
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        dialog.exec_()

    def ANOVA_cb(self):
        identifiers = list(self.editable_data.columns[2:])  # 从第三列开始提取物质名称

        numCols = self.tableWidget.columnCount()

        # 确保数据足够多
        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform ANOVA")
            return

        try:
            # 提取组别信息
            groups = [self.editable_data.iloc[i, 1] for i in range(self.editable_data.shape[0])]
            
            # 创建组别字典：每个组别的索引
            group_indices = {}
            for index, group in enumerate(groups):
                if group not in group_indices:
                    group_indices[group] = []
                group_indices[group].append(index)
            
            # 确保有至少两个不同组别
            if len(group_indices) < 2:
                QMessageBox.warning(self.tableWidget, "Error", "At least two distinct groups are required for the calculation")
                return

            unique_groups = list(group_indices.keys())
            all_means = {group: [] for group in unique_groups}
            all_stds = {group: [] for group in unique_groups}
            p_values = []

            # 对每个物质（列）计算ANOVA
            for col in range(2, numCols):
                group_data = []
                for group in unique_groups:
                    data = [float(self.tableWidget.item(row, col).text()) for row in group_indices[group]]
                    group_data.append(data)
                    mean, std = self.calculate_mean_std(data)
                    all_means[group].append(mean)
                    all_stds[group].append(std)

                # 执行单因素ANOVA
                f_stat, p_value = f_oneway(*group_data)
                p_values.append(p_value)

            # 显示结果
            self.display_anova_result(identifiers, all_means, all_stds, p_values)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def display_anova_result(self, identifiers, all_means, all_stds, p_values):
        unique_groups = list(all_means.keys())
        num_groups = len(unique_groups)
        
        # 动态设置列数，根据组数而定
        self.tableWidget_2.setRowCount(len(p_values))
        self.tableWidget_2.setColumnCount(2 + num_groups * 2)  # 每组2列：均值和标准差

        # 设置表头
        headers = ["Substance"]
        for group in unique_groups:
            headers += [f"Mean({group})", f"Std Dev ({group})"]
        headers.append("p Value")
        self.tableWidget_2.setHorizontalHeaderLabels(headers)

        # 填充数据
        for row, identifier in enumerate(identifiers):
            self.tableWidget_2.setItem(row, 0, QTableWidgetItem(str(identifier)))
            
            col_idx = 1
            for group in unique_groups:
                mean_item = QTableWidgetItem(f"{all_means[group][row]:.2f}")
                std_item = QTableWidgetItem(f"{all_stds[group][row]:.2f}")
                self.tableWidget_2.setItem(row, col_idx, mean_item)
                self.tableWidget_2.setItem(row, col_idx + 1, std_item)
                col_idx += 2

            # 填充 p 值
            p_value_item = QTableWidgetItem(f"{p_values[row]:.4f}")
            self.tableWidget_2.setItem(row, col_idx, p_value_item)

            # 自动调整行高
            self.tableWidget_2.resizeRowToContents(row)

        # 自动调整列宽
        for col in range(2 + num_groups * 2):
            self.tableWidget_2.resizeColumnToContents(col)

        QMessageBox.information(self.tableWidget_2, "Success", "ANOVA results have been updated")
        
    def Correlations_cb(self):
        # 从 tableWidget 中提取物质名称
        identifiers = list(self.editable_data.columns[2:])
        
        groups = [self.editable_data.iloc[i, 1] for i in range(self.editable_data.shape[0])]
        
        numCols = self.tableWidget.columnCount()

        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform correlation analysis")
            return

        try:
            # 创建组别的字典
            group_indices = {}
            for index, group in enumerate(groups):
                if group not in group_indices:
                    group_indices[group] = []
                group_indices[group].append(index) 

            # 统计不同组的数量
            num_groups = len(group_indices)

            # 若组数不等于2，给出提示并仅绘制热图
            if num_groups != 2:
                QMessageBox.warning(self.tableWidget_2, "Error", "More than two groups detected. Only the correlation heatmap will be generated.")
                self.plot_correlation_heatmap()
                return

            # 获取两个组别的索引
            unique_groups = list(group_indices.keys())
            group1_indices = group_indices[unique_groups[0]]
            group2_indices = group_indices[unique_groups[1]]

            # 检查样本数量是否相等
            if len(group1_indices) != len(group2_indices):
                QMessageBox.warning(self.tableWidget_2, "Error", "Unequal sample sizes between groups. Only the correlation heatmap will be generated.")
                self.plot_correlation_heatmap()
                return

            # 定义相关性分析结果存储
            correlation_results = []

            # 计算各物质在两组之间的相关性
            for col in range(2, numCols):  # 从第三列开始提取物质数据
                group1_data, group2_data = [], []
                for row in group1_indices:
                    item = self.tableWidget.item(row, col)
                    if item and item.text().strip():  # 确保单元格不为空
                        try:
                            group1_data.append(float(item.text()))
                        except ValueError:
                            print(f"Unable to convert the value at row {row}, column {col} in Group 1: '{item.text()}'")
                            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")
                            return
                
                for row in group2_indices:
                    item = self.tableWidget.item(row, col)
                    if item and item.text().strip():  # 确保单元格不为空
                        try:
                            group2_data.append(float(item.text()))
                        except ValueError:
                            print(f"Unable to convert the value at row {row}, column {col} in Group 2: '{item.text()}'")
                            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")
                            return       

                # 计算相关性
                if len(group1_data) > 1 and len(group2_data) > 1:
                    correlation, p_value = pearsonr(group1_data, group2_data)
                else:
                    correlation, p_value = None, None  # 数据不足

                # 保存结果
                correlation_results.append((identifiers[col - 2], correlation, p_value))
            
            # 显示结果
            self.display_correlation_results(correlation_results)

            # 绘制完整的分析（散点图 + 热图）
            self.plot_correlation_with_tooltip(correlation_results)

        except ValueError as e:
            print(f"Data format error: {str(e)}")
            QMessageBox.warning(self.tableWidget_2, "Error", "Invalid data format. Ensure all data are numeric")

    def display_correlation_results(self, correlation_results):
        # 清空 tableWidget_2
        self.tableWidget_2.clear()

        # 设置 tableWidget_2 的行数和列数
        self.tableWidget_2.setRowCount(len(correlation_results))
        self.tableWidget_2.setColumnCount(3)
        self.tableWidget_2.setHorizontalHeaderLabels(["Substance", "Correlation Coefficient", "P Value"])

        # 填充相关系数和 p 值到 tableWidget_2
        for row, (identifier, correlation, p_value) in enumerate(correlation_results):
            # 物质名称
            item_id = QTableWidgetItem(str(identifier))
            self.tableWidget_2.setItem(row, 0, item_id)

            # 相关系数
            item_corr = QTableWidgetItem(f"{correlation:.2f}" if correlation is not None else "N/A")
            self.tableWidget_2.setItem(row, 1, item_corr)

            # P 值
            item_p = QTableWidgetItem(f"{p_value:.4f}" if p_value is not None else "N/A")
            self.tableWidget_2.setItem(row, 2, item_p)

            # 调整每行的高度以适应内容
            self.tableWidget_2.resizeRowToContents(row)

        # 调整列宽适应内容
        for col in range(3):
            self.tableWidget_2.resizeColumnToContents(col)

        QMessageBox.information(self.tableWidget_2, "Success", "Correlation analysis results have been updated")

    def plot_correlation_with_tooltip(self, correlation_results, show_heatmap=True):
        # 过滤掉 None 值并保持索引匹配
        filtered_results = [(id, corr, p) for id, corr, p in correlation_results if corr is not None and p is not None]
        
        identifiers = [result[0] for result in filtered_results]
        correlations = [result[1] for result in filtered_results]
        p_values = [result[2] for result in filtered_results]

        # 创建散点图
        fig, ax = plt.subplots()
        scatter = ax.scatter(correlations, p_values, picker=True)

        # 设置标题和标签
        ax.set_title("Correlation Analysis")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_ylabel("P-value")

        # 创建悬停注释框
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            """更新注释框内容"""
            index = ind["ind"][0]  # 取第一个匹配的点
            x, y = scatter.get_offsets()[index]
            annot.xy = (x, y)
            identifier = identifiers[index]
            annot.set_text(f"{identifier}\nCorr: {x:.2f}\nP-val: {y:.4f}")
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            """鼠标悬停事件"""
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        # 绑定鼠标移动事件
        fig.canvas.mpl_connect("motion_notify_event", hover)

        
        self.plot_correlation_heatmap()
        
        plt.show()

    def plot_correlation_heatmap(self):
        """绘制带鼠标悬停功能的相关性热图"""
        df_data = self.original_data.iloc[1:, 2:]
        if not isinstance(df_data, pd.DataFrame):
            df_data = pd.DataFrame(df_data)
        
        # 确保所有列为数值型
        df_data = df_data.apply(pd.to_numeric, errors="coerce")
        
        # 计算相关性矩阵
        correlation_matrix = df_data.corr()
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 使用 imshow 绘制热图（直接控制 QuadMesh）
        im = ax.imshow(
            correlation_matrix,
            cmap="coolwarm",
            aspect="auto",
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )
        im.set_picker(True)  # 关键修复：强制启用拾取
        
        # 设置坐标轴标签
        x_labels = correlation_matrix.columns.tolist()
        y_labels = correlation_matrix.index.tolist()
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticklabels(y_labels)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 绑定悬停事件
        cursor = mplcursors.cursor(im, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
            """鼠标悬停时显示变量名和相关系数"""
            row_idx, col_idx = int(sel.target[1]), int(sel.target[0])  # imshow 的坐标顺序为 (x, y)
            if 0 <= row_idx < len(y_labels) and 0 <= col_idx < len(x_labels):
                x_label = x_labels[col_idx]
                y_label = y_labels[row_idx]
                corr_value = correlation_matrix.iloc[row_idx, col_idx]
                
                sel.annotation.set_text(f"Row: {x_label}\nCol: {y_label}\nCorrelation: {corr_value:.2f}")
                sel.annotation.get_bbox_patch().set_alpha(0.8)

        # 手动实现 keep_inside 功能
        def on_move(event):
            """鼠标移动时检查是否在热图内"""
            if not event.inaxes:  # 如果鼠标不在热图内
                for sel in cursor.selections:
                    sel.annotation.set_visible(False)
                plt.draw()

        # 绑定鼠标移动事件
        fig.canvas.mpl_connect("motion_notify_event", on_move)

        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def PCA_cb(self):
        # 清空 tableWidget2
        self.tableWidget_2.clear()

        data = []
        labels = []  # 用于存储标签
        numRows = self.tableWidget.rowCount()
        numCols = self.tableWidget.columnCount()

        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform PCA analysis")
            return

        try:
            for row in range(numRows):
                row_data = []
                for col in range(2, numCols):  
                    row_data.append(float(self.tableWidget.item(row, col).text()))
                data.append(row_data)
                # 获取标签数据（假设在第二列存放组别信息，0/1表示不同组别）
                labels.append(self.editable_data.iloc[row, 1])
 
 
            data = np.array(data)
            labels = np.array(labels)

            # 数据标准化
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # PCA降维
            self.pca = PCA(n_components=5)  # 取前5个主成分
            pca_result = self.pca.fit_transform(data_scaled)
            
            # 将PCA结果转为DataFrame并添加标签列
            pca_df = pd.DataFrame(pca_result, columns=[f"PC {i+1}" for i in range(5)])
            pca_df['Label'] = labels  # 标签列
            pca_df['Type'] = 'Score'  # 添加标识列：得分矩阵
            pca_df.index = [f"Sample_{i+1}" for i in range(pca_df.shape[0])]  # 修改行索引

            # 获取载荷矩阵
            loadings = self.pca.components_.T  # 转置以匹配原始变量
            loadings_df = pd.DataFrame(loadings, columns=[f"PC {i+1}" for i in range(5)], 
                                    index=[f"Var {i+1}" for i in range(data.shape[1])])
            loadings_df['Type'] = 'Loading'  # 添加标识列：载荷矩阵
            loadings_df['Label'] = 'N/A'     # 载荷矩阵无标签，填充占位符

            # 将得分矩阵和载荷矩阵的数值精确到2位有效数字
            pca_df = pca_df.round(4)  # 得分矩阵
            loadings_df = loadings_df.round(4)  # 载荷矩阵

            # 将得分矩阵和载荷矩阵合并为一个表格
            combined_df = pd.concat([pca_df, loadings_df], axis=0)
            columns = ['Type', 'Label'] + [f"PC {i+1}" for i in range(5)]  # 将标识列放在前两列
            combined_df = combined_df[columns]  

            # 将合并后的表格显示在 tableWidget_2 中
            self.display_table(combined_df)

            # 使用Seaborn绘制多维散点矩阵
            self.visualize_pca_pairplot(pca_df)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def display_table(self, df):
        """将 DataFrame 显示在 tableWidget_2 中"""
        self.tableWidget_2.setRowCount(df.shape[0])
        self.tableWidget_2.setColumnCount(df.shape[1])
        self.tableWidget_2.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                self.tableWidget_2.setItem(i, j, item)

    def visualize_pca_pairplot(self, pca_df):
        sns.set(style="whitegrid")

        # 获取 PCA 方差贡献率
        explained_variance = self.pca.explained_variance_ratio_  # 获取方差贡献率

        # 自定义对角线绘制函数，添加方差贡献率文本
        def add_variance_text(x, **kwargs):
            ax = plt.gca()  # 获取当前子图
            pc_index = int(x.name.split()[1]) - 1  # 获取主成分索引
            variance_text = f"PC {pc_index+1}\n{explained_variance[pc_index] * 100:.1f}%"  # 格式化文本
            ax.text(0.5, 0.5, variance_text, ha="center", va="center", fontsize=14, fontweight="bold")  # 居中显示文本

        # 绘制 PairPlot 并替换对角线
        pairplot = sns.pairplot(
            pca_df, 
            hue="Label", 
            plot_kws={'alpha': 0.6, 's': 50}, 
            diag_kind="hist"
        )
        pairplot.map_diag(add_variance_text)  # 让对角线显示方差贡献率

        pairplot.fig.suptitle("PCA Pair Plot", y=1.02)  # 设置标题
        plt.show()

        QMessageBox.information(self.tableWidget_2, "Success", "PCA multidimensional scatter plot has been generated")
   
    def PLS_cb(self):
        # 清空 tableWidget_2
        self.tableWidget_2.clear()

        data = []
        labels = []  # 存储标签（分类变量）
        numRows = self.tableWidget.rowCount()
        numCols = self.tableWidget.columnCount()

        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform PLS analysis")
            return

        try:
            for row in range(numRows):
                row_data = []
                for col in range(2, numCols):  
                    row_data.append(float(self.tableWidget.item(row, col).text()))
                data.append(row_data)
                # 获取分类标签
                labels.append(self.editable_data.iloc[row, 1])

            data = np.array(data)
            labels = np.array(labels).reshape(-1, 1)  # PLS需要二维标签

            # 数据标准化
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            labels_scaled = StandardScaler().fit_transform(labels)  # 标签也进行标准化

            # PLS 分析
            self.pls = PLSRegression(n_components=5)  # 取前5个成分
            pls_result = self.pls.fit_transform(data_scaled, labels_scaled)[0]
            
            # 将PLS得分转为DataFrame并添加标签列
            pls_df = pd.DataFrame(pls_result, columns=[f"PLS {i+1}" for i in range(5)])
            pls_df['Label'] = labels.flatten()  # 标签列
            pls_df['Type'] = 'Score'  # 标识得分矩阵
            pls_df.index = [f"Sample_{i+1}" for i in range(pls_df.shape[0])]  # 修改行索引

            # 获取载荷矩阵
            loadings = self.pls.x_weights_  # PLS载荷矩阵
            loadings_df = pd.DataFrame(loadings, columns=[f"PLS {i+1}" for i in range(5)],
                                    index=[f"Var {i+1}" for i in range(data.shape[1])])
            loadings_df['Type'] = 'Loading'  # 标识载荷矩阵
            loadings_df['Label'] = 'N/A'  # 载荷矩阵无标签，填充占位符

            # 结果数值四舍五入到4位小数
            pls_df = pls_df.round(4)
            loadings_df = loadings_df.round(4)

            # 合并得分矩阵和载荷矩阵
            combined_df = pd.concat([pls_df, loadings_df], axis=0)
            columns = ['Type', 'Label'] + [f"PLS {i+1}" for i in range(5)]  # 重新排列列顺序
            combined_df = combined_df[columns]

            # 将结果显示在 tableWidget_2 中
            self.display_table(combined_df)

            # 绘制 PLS 结果可视化
            self.visualize_pls_pairplot(pls_df)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def visualize_pls_pairplot(self, pls_df):
        sns.set(style="whitegrid")
        
        # 绘制 PLS 多维散点矩阵
        pairplot = sns.pairplot(
            pls_df, 
            hue="Label", 
            plot_kws={'alpha': 0.6, 's': 50},
            diag_kind="hist"
        )
        
        pairplot.fig.suptitle("PLS Pair Plot", y=1.02)  # 设置标题
        plt.show()
        
        QMessageBox.information(self.tableWidget_2, "Success", "PLS multidimensional scatter plot has been generated")

    def Factor_Analysis_cb(self):
        # 清空 tableWidget2
        self.tableWidget_2.clear()

        # 提取数据
        data = []
        identifiers = list(self.original_data.columns[2:])
        numRows = self.tableWidget.rowCount()
        numCols = self.tableWidget.columnCount()

        if numCols < 2:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform factor analysis")
            return

        try:
            # 提取所有列的数据
            for row in range(numRows):
                row_data = []
                for col in range(2, numCols):  
                    row_data.append(float(self.tableWidget.item(row, col).text()))
                data.append(row_data)

            # 标准化数据
            data = StandardScaler().fit_transform(data)

            # KMO 检验
            kmo_all, kmo_model = calculate_kmo(np.array(data))
            if kmo_model < 0.6:
                QMessageBox.warning(self.tableWidget, "Warning", "KMO test value is below 0.6, unsuitable for factor analysis")
                return

            # Bartlett 球形度检验
            chi_square_value, p_value = calculate_bartlett_sphericity(np.array(data))
            if p_value > 0.05:
                QMessageBox.warning(self.tableWidget, "Warning", "Bartlett's sphericity test is not significant; data may be unsuitable for factor analysis")
                return

            # 使用PCA进行初步因子分析
            pca = PCA()
            pca.fit(data)

            # 获取方差贡献率
            explained_variance_ratio = pca.explained_variance_ratio_

            # 绘制碎石图
            self.plot_scree_plot(explained_variance_ratio)

            # 获取因子载荷矩阵
            loadings = pca.components_.T

            # 旋转因子载荷矩阵（Varimax旋转）
            fa = FactorAnalyzer(rotation='varimax', n_factors=len(explained_variance_ratio))
            fa.fit(data)
            rotated_loadings = fa.loadings_

            # 展示结果，包括旋转前和旋转后的因子载荷矩阵
            self.display_factor_analysis_result(identifiers, explained_variance_ratio, loadings, rotated_loadings, kmo_model, chi_square_value, p_value)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def plot_scree_plot(self, explained_variance_ratio):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
        plt.xlabel('Number of Factors')
        plt.ylabel('Variance Contribution Rate')
        plt.title('Scree Plot')
        plt.show()

    def display_factor_analysis_result(self, identifiers, explained_variance_ratio, loadings, rotated_loadings, kmo, chi_square_value, p_value):
        # 清空 tableWidget2
        self.tableWidget_2.clear()

        # 设置表格行数和列数
        num_factors = loadings.shape[1]
        self.tableWidget_2.setRowCount(len(identifiers) + 2)  # 多两行用于放 KMO 和 Bartlett 检验结果
        self.tableWidget_2.setColumnCount(num_factors + 3)  # 因子载荷、旋转因子载荷和共同性
        headers = ["Substance"] + [f"Factor {i+1}" for i in range(num_factors)] + ["Communality"]
        self.tableWidget_2.setHorizontalHeaderLabels(headers)

        # 填充表格
        for row, (identifier, loading, rotated_loading) in enumerate(zip(identifiers, loadings, rotated_loadings)):
            item_id = QTableWidgetItem(str(identifier))
            self.tableWidget_2.setItem(row, 0, item_id)

            for col in range(num_factors):
                # 显示旋转前的因子载荷
                item_loading = QTableWidgetItem(f"{loading[col]:.2f}")
                self.tableWidget_2.setItem(row, col + 1, item_loading)

                # 显示旋转后的因子载荷
                item_rotated_loading = QTableWidgetItem(f"{rotated_loading[col]:.2f}")
                self.tableWidget_2.setItem(row, col + 1, item_rotated_loading)

            # 计算并显示共同性
            communality = sum(rotated_loading**2)
            item_communality = QTableWidgetItem(f"{communality:.2f}")
            self.tableWidget_2.setItem(row, num_factors + 1, item_communality)

        # 添加方差解释率在最后一行
        self.tableWidget_2.insertRow(len(identifiers))
        self.tableWidget_2.setItem(len(identifiers), 0, QTableWidgetItem("Variance Explained Ratio"))
        for col, var in enumerate(explained_variance_ratio):
            self.tableWidget_2.setItem(len(identifiers), col + 1, QTableWidgetItem(f"{var:.2f}"))

        # 添加 KMO 和 Bartlett 检验结果
        self.tableWidget_2.setItem(len(identifiers) + 1, 0, QTableWidgetItem("KMO Test"))
        self.tableWidget_2.setItem(len(identifiers) + 1, 1, QTableWidgetItem(f"{kmo:.2f}"))

        self.tableWidget_2.setItem(len(identifiers) + 2, 0, QTableWidgetItem("Bartlett's Test of Sphericity"))
        self.tableWidget_2.setItem(len(identifiers) + 2, 1, QTableWidgetItem(f"χ²: {chi_square_value:.2f}, p: {p_value:.3f}"))

        # 调整列宽适合内容
        for col in range(num_factors + 2):
            self.tableWidget_2.resizeColumnToContents(col)
        
         # 弹出提示框解释 KMO 和 Bartlett 标准
        QMessageBox.information(self.tableWidget_2, "KMO and Bartlett Test Standards", 
                                "KMO value above 0.5 is suitable for factor analysis. Bartlett's test p-value should typically be less than 0.05 to qualify for factor analysis.")

        QMessageBox.information(self.tableWidget_2, "Success", "Factor analysis results have been updated")

    def Discriminant_Analysis_cb(self):
        # 清空 tableWidget2
        self.tableWidget_2.clear()

        # 提取数据和标签
        data = []
        labels = []
        identifiers = list(self.editable_data.columns[2:])
        numRows = self.tableWidget.rowCount()
        numCols = self.tableWidget.columnCount()

        if numCols < 3:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform discriminant analysis")
            return

        try:
            for row in range(numRows):
                row_data = []
                for col in range(2, numCols):
                    row_data.append(float(self.tableWidget.item(row, col).text()))
                data.append(row_data)
                # 假设标签在第二列
                labels.append(self.editable_data.iloc[row, 1])
            
            # 执行线性判别分析（LDA）
            lda = LinearDiscriminantAnalysis()
            data = np.array(data)
            lda.fit(data, labels)
            transformed_data = lda.transform(data)
            explained_variance_ratio = lda.explained_variance_ratio_

            # 绘制每个判别变量的解释方差比
            self.plot_discriminant_variance(explained_variance_ratio)

            # 显示结果
            self.display_discriminant_analysis_result(identifiers, explained_variance_ratio, transformed_data)

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def plot_discriminant_variance(self, explained_variance_ratio):
        plt.figure(figsize=(8, 5))
        plt.tick_params(axis='both', labelsize=12)  # 12 是示例字号，按需调整
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='skyblue')
        plt.xlabel('Discriminant Function')
        plt.ylabel('Variance Contribution Rate')
        plt.title('Discriminant Analysis Variance Contribution Plot')
        plt.show()

    def display_discriminant_analysis_result(self, identifiers, explained_variance_ratio, transformed_data):
        # 清空 tableWidget_2
        self.tableWidget_2.clear()

        num_discriminants = transformed_data.shape[1]
        self.tableWidget_2.setRowCount(len(identifiers))
        self.tableWidget_2.setColumnCount(num_discriminants + 1)
        headers = ["Substance"] + [f"Discriminant Function {i+1}" for i in range(num_discriminants)]
        self.tableWidget_2.setHorizontalHeaderLabels(headers)

        # 将转换后的数据填入表格
        for row, (identifier, transformed_row) in enumerate(zip(identifiers, transformed_data)):
            item_id = QTableWidgetItem(str(identifier))
            self.tableWidget_2.setItem(row, 0, item_id)

            for col in range(num_discriminants):
                item_transform = QTableWidgetItem(f"{transformed_row[col]:.2f}")
                self.tableWidget_2.setItem(row, col + 1, item_transform)

        # 在最后一行插入解释方差
        self.tableWidget_2.insertRow(len(identifiers))
        self.tableWidget_2.setItem(len(identifiers), 0, QTableWidgetItem("Variance Explained Ratio"))
        for col, var in enumerate(explained_variance_ratio):
            self.tableWidget_2.setItem(len(identifiers), col + 1, QTableWidgetItem(f"{var:.2f}"))

        # 调整列宽以适应内容
        for col in range(num_discriminants + 1):
            self.tableWidget_2.resizeColumnToContents(col)

        QMessageBox.information(self.tableWidget_2, "Success", "Discriminant analysis results have been updated")

    def Cluster_Analysis_cb(self):
        # 清空 tableWidget_2
        self.tableWidget_2.clear()
        
        # 提取数据
        data = []
        identifiers = list(self.original_data.columns[2:])
        numRows = self.tableWidget.rowCount()
        numCols = self.tableWidget.columnCount()
        
        if numCols < 3:
            QMessageBox.warning(self.tableWidget, "Error", "Insufficient columns to perform clustering analysis")
            return

        try:
            for row in range(numRows):
                row_data = []
                for col in range(2, numCols):
                    row_data.append(float(self.tableWidget.item(row, col).text()))
                data.append(row_data)

            # K-Means 聚类
            num_clusters = 3  # 设置聚类数
            data = np.array(data)
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            kmeans.fit(data)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            
            # 显示聚类分析结果
            self.display_cluster_analysis_result(identifiers, cluster_labels, cluster_centers)

            # 绘制聚类 PCA 图
            self.plot_cluster_pca(data, cluster_labels)

            # 绘制聚类热图
            self.plot_cluster_heatmap(data, cluster_labels,identifiers)
            

        except ValueError:
            QMessageBox.warning(self.tableWidget, "Error", "Invalid data format. Ensure all data are numeric")

    def plot_cluster_pca(self, data, cluster_labels):
        """
        绘制 K-Means PCA 聚类图
        """
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        df_pca = np.column_stack((data_pca, cluster_labels))
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=cluster_labels, palette='Set1', s=100, edgecolor='black')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
        plt.title("K-Means Clustering with PCA")
        plt.legend(title="Cluster")
        plt.show()

    def plot_cluster_heatmap(self, data, cluster_labels, identifiers):
        """
        绘制 K-Means 聚类热图，添加 Z-score 计算，并在左侧标注物质名称
        """
        # 计算 Z-score 进行标准化
        data_zscore = zscore(data, axis=0)  # 按行（物质）计算 Z-score

        # 转换为 DataFrame，行索引为物质名称，列索引为样本编号
        df = pd.DataFrame(data_zscore, columns=identifiers)

        # 转置 DataFrame，确保物质是行索引
        df = df.T  

        # 设置 X 轴为分组（假设 cluster_labels 对应列编号）
        group_labels = [f"Sample {i+1}" for i in range(df.shape[1])]
        df.columns = group_labels  # 设定列索引

        # 层次聚类链接
        row_linkage = linkage(df, method='ward')

        # 画出 clustermap
        g = sns.clustermap(
            df, 
            row_cluster=True, col_cluster=False,  # 仅对物质进行聚类
            row_linkage=row_linkage, 
            cmap="RdYlGn", 
            linewidths=0.5, 
            figsize=(12, 8), 
            z_score=None,  # 已经手动计算了 Z-score
            yticklabels=True,
            cbar_kws={"shrink": 0.5}, # 缩小颜色条，避免影响图像
            dendrogram_ratio=(0.02, 0.1), # 调整左侧树状图和顶部空白区域大小
           
        )

        # 调整热图边距，防止边界裁切
        g.ax_heatmap.set_position([0.2, 0.1, 0.6, 0.8])  # [left, bottom, width, height]

        # 调整整个图的边距
        g.fig.subplots_adjust(left=0.06, right=0.9, top=0.95, bottom=0.15)

        # 设置轴标签
        g.ax_heatmap.set_xlabel("Samples", fontsize=12)
        g.ax_heatmap.set_ylabel("Substances", fontsize=12)

        plt.title("Clustered Heatmap\nwith Z-score", fontsize=12)
        plt.show()

    def display_cluster_analysis_result(self, identifiers, cluster_labels, cluster_centers):
        # Clear tableWidget_2
        self.tableWidget_2.clear()

        num_clusters = cluster_centers.shape[0]
        self.tableWidget_2.setRowCount(len(identifiers))
        self.tableWidget_2.setColumnCount(2 + num_clusters)  # 2 for ID and Cluster, plus cluster centers
        headers = ["Substance", "Cluster Label"] + [f"Cluster Center {i+1}" for i in range(num_clusters)]
        self.tableWidget_2.setHorizontalHeaderLabels(headers)

        # Fill table with clustering results
        for row, (identifier, cluster_label) in enumerate(zip(identifiers, cluster_labels)):
            item_id = QTableWidgetItem(str(identifier))
            self.tableWidget_2.setItem(row, 0, item_id)

            item_cluster = QTableWidgetItem(str(cluster_label))
            self.tableWidget_2.setItem(row, 1, item_cluster)

        # Display cluster centers in the last row
        self.tableWidget_2.insertRow(len(identifiers))
        self.tableWidget_2.setItem(len(identifiers), 0, QTableWidgetItem("Cluster Center"))

        for i, center in enumerate(cluster_centers):
            for j, value in enumerate(center):
                item_center = QTableWidgetItem(f"{value:.2f}")
                self.tableWidget_2.setItem(len(identifiers), j + 2, item_center)

        # Adjust column widths to fit content
        for col in range(2 + num_clusters):
            self.tableWidget_2.resizeColumnToContents(col)

        QMessageBox.information(self.tableWidget_2, "Success", "Clustering analysis results have been updated")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FirstWindow()
    window.show()
    sys.exit(app.exec_())