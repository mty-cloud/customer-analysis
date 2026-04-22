"""
customer_analysis_app.py
客户价值分析可视化系统 - 基于RFM聚类
符合数据可视化课程设计要求
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CustomerAnalysisApp:
    """客户价值分析可视化应用"""

    def __init__(self, root):
        self.root = root
        self.root.title("客户价值分析可视化系统")
        self.root.geometry("1400x900")

        # 数据存储
        self.data = None
        self.rfm_data = None
        self.cluster_result = None

        # 设置样式
        self.setup_styles()

        # 创建界面
        self.create_widgets()

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('微软雅黑', 16, 'bold'))
        style.configure('Header.TLabel', font=('微软雅黑', 12, 'bold'))
        style.configure('Normal.TLabel', font=('微软雅黑', 10))

    def create_widgets(self):
        """创建界面组件"""
        # 顶部标题
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(title_frame, text="客户价值分析可视化系统",
                  style='Title.TLabel').pack()
        ttk.Label(title_frame, text="基于RFM模型的K-Means聚类分析",
                  style='Normal.TLabel').pack()

        # 主内容区域
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧控制面板
        self.create_control_panel(main_frame)

        # 右侧可视化区域
        self.create_visualization_panel(main_frame)

        # 底部状态栏
        self.create_status_bar()

    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 数据加载区域
        load_frame = ttk.LabelFrame(control_frame, text="数据加载")
        load_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(load_frame, text="加载数据文件",
                   command=self.load_data).pack(fill=tk.X, padx=10, pady=5)

        self.file_label = ttk.Label(load_frame, text="未加载文件",
                                    style='Normal.TLabel')
        self.file_label.pack(padx=10, pady=5)

        # 数据信息显示
        info_frame = ttk.LabelFrame(control_frame, text="数据信息")
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        self.info_text = tk.Text(info_frame, height=8, width=40,
                                 font=('微软雅黑', 9))
        self.info_text.pack(padx=10, pady=5)

        # 分析参数设置
        param_frame = ttk.LabelFrame(control_frame, text="聚类参数设置")
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(param_frame, text="聚类数量 K:").pack(anchor=tk.W, padx=10, pady=(5, 0))

        self.k_var = tk.StringVar(value="4")
        k_spinbox = ttk.Spinbox(param_frame, from_=2, to=8,
                                textvariable=self.k_var, width=10)
        k_spinbox.pack(anchor=tk.W, padx=10, pady=5)

        # 分析按钮
        ttk.Button(param_frame, text="执行聚类分析",
                   command=self.run_clustering).pack(fill=tk.X, padx=10, pady=10)

        # 可视化选项
        viz_frame = ttk.LabelFrame(control_frame, text="可视化选项")
        viz_frame.pack(fill=tk.X, padx=10, pady=10)

        self.viz_var = tk.StringVar(value="cluster_scatter")

        viz_options = [
            ("聚类散点图", "cluster_scatter"),
            ("RFM分布图", "rfm_distribution"),
            ("肘部法则图", "elbow_plot"),
            ("客户画像雷达图", "radar_plot"),
            ("客户价值矩阵", "value_matrix")
        ]

        for text, value in viz_options:
            ttk.Radiobutton(viz_frame, text=text, variable=self.viz_var,
                            value=value, command=self.update_visualization
                            ).pack(anchor=tk.W, padx=20, pady=2)

        # 报告生成
        report_frame = ttk.LabelFrame(control_frame, text="分析报告")
        report_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(report_frame, text="生成分析报告",
                   command=self.generate_report).pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(report_frame, text="导出报告",
                   command=self.export_report).pack(fill=tk.X, padx=10, pady=5)

    def create_visualization_panel(self, parent):
        """创建右侧可视化面板"""
        viz_frame = ttk.LabelFrame(parent, text="可视化展示")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建notebook用于多图表展示
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 主图表标签页
        self.main_chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_chart_frame, text="主图表")

        # 数据预览标签页
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")

        # 分析结论标签页
        self.conclusion_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.conclusion_frame, text="分析结论")

        # 创建数据预览表格
        self.create_preview_table()

        # 创建结论文本框
        self.create_conclusion_text()

    def create_preview_table(self):
        """创建数据预览表格"""
        # 创建滚动条
        scrollbar_y = ttk.Scrollbar(self.preview_frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = ttk.Scrollbar(self.preview_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # 创建表格
        self.tree = ttk.Treeview(self.preview_frame,
                                 yscrollcommand=scrollbar_y.set,
                                 xscrollcommand=scrollbar_x.set)

        scrollbar_y.config(command=self.tree.yview)
        scrollbar_x.config(command=self.tree.xview)

        self.tree.pack(fill=tk.BOTH, expand=True)

    def create_conclusion_text(self):
        """创建结论文本框"""
        text_frame = ttk.Frame(self.conclusion_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.conclusion_text = tk.Text(text_frame, wrap=tk.WORD,
                                       font=('微软雅黑', 11))
        self.conclusion_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.conclusion_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.conclusion_text.config(yscrollcommand=scrollbar.set)

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_data(self):
        """加载数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"),
                       ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            # 根据文件扩展名加载数据
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.txt'):
                self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            else:
                messagebox.showerror("错误", "不支持的文件格式")
                return

            # 更新界面
            filename = file_path.split('/')[-1]
            self.file_label.config(text=f"已加载: {filename}")

            # 更新数据信息
            self.update_data_info()

            # 更新预览表格
            self.update_preview_table()

            # 自动计算RFM
            self.calculate_rfm()

            self.status_bar.config(text=f"成功加载数据：{len(self.data)}行 × {len(self.data.columns)}列")

        except Exception as e:
            messagebox.showerror("加载失败", f"加载数据时出错：{str(e)}")

    def update_data_info(self):
        """更新数据信息显示"""
        self.info_text.delete(1.0, tk.END)

        if self.data is None:
            return

        info = f"数据规模：{len(self.data)}行 × {len(self.data.columns)}列\n"
        info += f"缺失值数量：{self.data.isnull().sum().sum()}\n\n"
        info += "数值列统计：\n"

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # 最多显示5列
            info += f"  {col}: {self.data[col].mean():.2f} (均值)\n"

        self.info_text.insert(1.0, info)

    def update_preview_table(self):
        """更新数据预览表格"""
        # 清空现有内容
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.data is None:
            return

        # 设置列
        columns = list(self.data.columns[:10])  # 最多显示10列
        self.tree['columns'] = columns
        self.tree['show'] = 'headings'

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # 添加数据（前20行）
        for i, row in self.data.head(20).iterrows():
            values = [str(row[col])[:20] for col in columns]
            self.tree.insert('', 'end', values=values)

    def calculate_rfm(self):
        """计算RFM指标"""
        if self.data is None:
            return

        try:
            # 识别RFM相关列
            rfm_df = pd.DataFrame()

            # 尝试找到合适的列
            if '最近消费日期' in self.data.columns:
                # 计算Recency（距今天数）
                latest_date = pd.to_datetime(self.data['最近消费日期']).max()
                rfm_df['Recency'] = (latest_date - pd.to_datetime(self.data['最近消费日期'])).dt.days
            elif '消费频率' in self.data.columns:
                rfm_df['Frequency'] = self.data['消费频率']
            elif '总消费金额' in self.data.columns:
                rfm_df['Monetary'] = self.data['总消费金额']
            else:
                # 如果没有标准列，尝试自动识别
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    if '频率' in col or '次数' in col:
                        rfm_df['Frequency'] = self.data[col]
                    elif '金额' in col or '消费' in col:
                        rfm_df['Monetary'] = self.data[col]

            self.rfm_data = rfm_df
            self.status_bar.config(text="RFM指标计算完成")

        except Exception as e:
            self.status_bar.config(text=f"RFM计算失败：{str(e)}")

    def run_clustering(self):
        """执行聚类分析"""
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            k = int(self.k_var.get())

            # 准备聚类数据
            cluster_cols = []
            for col in self.data.select_dtypes(include=[np.number]).columns:
                if col not in ['客户ID', '客户类型标签'] and self.data[col].notna().sum() > 0:
                    cluster_cols.append(col)

            if len(cluster_cols) == 0:
                messagebox.showerror("错误", "没有可用于聚类的数值列")
                return

            # 选择关键特征
            key_features = []
            feature_priority = ['消费频率', '总消费金额', '会员年限', '平均单次消费',
                                '满意度评分', '折扣使用率', '年龄']

            for feature in feature_priority:
                if feature in cluster_cols:
                    key_features.append(feature)

            if len(key_features) < 2:
                key_features = cluster_cols[:5]

            # 数据预处理
            cluster_data = self.data[key_features].copy()
            cluster_data = cluster_data.fillna(cluster_data.median())

            # 标准化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            # K-Means聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.cluster_result = kmeans.fit_predict(scaled_data)
            self.data['聚类标签'] = self.cluster_result

            # 更新可视化
            self.update_visualization()

            # 生成分析结论
            self.generate_conclusion(key_features, kmeans)

            self.status_bar.config(text=f"聚类分析完成，将客户分为{k}个群体")

        except Exception as e:
            messagebox.showerror("聚类失败", f"执行聚类分析时出错：{str(e)}")

    def update_visualization(self):
        """更新可视化图表"""
        # 清除旧图表
        for widget in self.main_chart_frame.winfo_children():
            widget.destroy()

        if self.data is None or self.cluster_result is None:
            return

        viz_type = self.viz_var.get()

        if viz_type == "cluster_scatter":
            self.plot_cluster_scatter()
        elif viz_type == "rfm_distribution":
            self.plot_rfm_distribution()
        elif viz_type == "elbow_plot":
            self.plot_elbow()
        elif viz_type == "radar_plot":
            self.plot_radar()
        elif viz_type == "value_matrix":
            self.plot_value_matrix()

    def plot_cluster_scatter(self):
        """绘制聚类散点图"""
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # 选择两个特征绘制散点图
        x_col = '消费频率' if '消费频率' in self.data.columns else self.data.select_dtypes(include=[np.number]).columns[
            0]
        y_col = '总消费金额' if '总消费金额' in self.data.columns else \
        self.data.select_dtypes(include=[np.number]).columns[-1]

        scatter = ax.scatter(self.data[x_col], self.data[y_col],
                             c=self.cluster_result, cmap='viridis',
                             alpha=0.6, s=30)

        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f'客户聚类结果 ({len(set(self.cluster_result))}个群体)', fontsize=12)

        # 添加颜色条
        fig.colorbar(scatter, ax=ax, label='聚类标签')

        # 添加聚类中心
        centers_x = [self.data[self.data['聚类标签'] == i][x_col].mean()
                     for i in range(len(set(self.cluster_result)))]
        centers_y = [self.data[self.data['聚类标签'] == i][y_col].mean()
                     for i in range(len(set(self.cluster_result)))]
        ax.scatter(centers_x, centers_y, c='red', marker='X', s=200,
                   edgecolors='white', linewidth=2, label='聚类中心')

        ax.legend()
        ax.grid(True, alpha=0.3)

        self.display_figure(fig)

    def plot_rfm_distribution(self):
        """绘制RFM分布图"""
        fig = Figure(figsize=(12, 8), dpi=100)

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:3]

        for i, col in enumerate(numeric_cols, 1):
            ax = fig.add_subplot(1, 3, i)

            for cluster in sorted(set(self.cluster_result)):
                cluster_data = self.data[self.data['聚类标签'] == cluster][col].dropna()
                ax.hist(cluster_data, bins=20, alpha=0.5, label=f'群体{cluster}')

            ax.set_xlabel(col)
            ax.set_ylabel('频数')
            ax.set_title(f'{col}分布')
            ax.legend()

        fig.suptitle('各客户群体的特征分布', fontsize=14)
        fig.tight_layout()

        self.display_figure(fig)

    def plot_elbow(self):
        """绘制肘部法则图"""
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        # 准备数据
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        cluster_data = self.data[numeric_cols].fillna(self.data[numeric_cols].median())

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # 计算不同K值的SSE
        sse = []
        k_range = range(1, 10)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            sse.append(kmeans.inertia_)

        # 绘制肘部曲线
        ax.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('聚类数量 K', fontsize=11)
        ax.set_ylabel('SSE (簇内平方和)', fontsize=11)
        ax.set_title('肘部法则 - 最佳聚类数选择', fontsize=12)

        # 标注当前K值
        current_k = int(self.k_var.get())
        if current_k < len(sse):
            ax.plot(current_k, sse[current_k - 1], 'ro', markersize=12,
                    label=f'当前选择 K={current_k}')
            ax.legend()

        ax.grid(True, alpha=0.3)

        self.display_figure(fig)

    def plot_radar(self):
        """绘制客户画像雷达图"""
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='polar')

        # 选择用于雷达图的特征
        features = ['消费频率', '总消费金额', '会员年限', '平均单次消费', '满意度评分']
        available_features = [f for f in features if f in self.data.columns]

        if len(available_features) < 3:
            available_features = list(self.data.select_dtypes(include=[np.number]).columns[:5])

        # 计算每个聚类的特征均值并标准化
        cluster_profiles = []
        for cluster in sorted(set(self.cluster_result)):
            cluster_data = self.data[self.data['聚类标签'] == cluster][available_features]
            profile = cluster_data.mean().values

            # 归一化到0-1范围
            profile_min = profile.min()
            profile_max = profile.max()
            if profile_max > profile_min:
                profile = (profile - profile_min) / (profile_max - profile_min)
            cluster_profiles.append(profile)

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
        angles += angles[:1]

        # 绘制雷达图
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_profiles)))

        for i, profile in enumerate(cluster_profiles):
            values = profile.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i],
                    label=f'群体{i}')
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_features, fontsize=9)
        ax.set_title('各客户群体特征雷达图', fontsize=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        self.display_figure(fig)

    def plot_value_matrix(self):
        """绘制客户价值矩阵"""
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)

        # 计算每个聚类的平均价值和平均频率
        if '总消费金额' in self.data.columns and '消费频率' in self.data.columns:
            cluster_stats = self.data.groupby('聚类标签').agg({
                '消费频率': 'mean',
                '总消费金额': 'mean',
                '客户ID': 'count'
            }).rename(columns={'客户ID': '客户数量'})

            # 绘制气泡图
            scatter = ax.scatter(
                cluster_stats['消费频率'],
                cluster_stats['总消费金额'],
                s=cluster_stats['客户数量'] * 10,
                c=cluster_stats.index,
                cmap='viridis',
                alpha=0.6
            )

            # 添加标签
            for idx, row in cluster_stats.iterrows():
                ax.annotate(f'群体{idx}',
                            (row['消费频率'], row['总消费金额']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold')

            ax.set_xlabel('消费频率 (Frequency)', fontsize=11)
            ax.set_ylabel('总消费金额 (Monetary)', fontsize=11)
            ax.set_title('客户价值矩阵', fontsize=12)

            # 添加象限线
            ax.axhline(y=cluster_stats['总消费金额'].median(),
                       color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=cluster_stats['消费频率'].median(),
                       color='gray', linestyle='--', alpha=0.5)

            fig.colorbar(scatter, ax=ax, label='聚类标签')
            ax.grid(True, alpha=0.3)

        self.display_figure(fig)

    def display_figure(self, fig):
        """在界面中显示图表"""
        canvas = FigureCanvasTkAgg(fig, self.main_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_conclusion(self, features, kmeans):
        """生成分析结论"""
        self.conclusion_text.delete(1.0, tk.END)

        conclusion = "=" * 60 + "\n"
        conclusion += "                    客户价值分析报告\n"
        conclusion += "=" * 60 + "\n\n"

        # 数据概况
        conclusion += "【一、数据概况】\n"
        conclusion += "-" * 40 + "\n"
        conclusion += f"分析样本量：{len(self.data)} 位客户\n"
        conclusion += f"分析特征数：{len(features)} 个\n"
        conclusion += f"聚类数量：{len(set(self.cluster_result))} 个客户群体\n\n"

        # 聚类结果统计
        conclusion += "【二、客户群体分布】\n"
        conclusion += "-" * 40 + "\n"

        cluster_counts = self.data['聚类标签'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = count / len(self.data) * 100
            conclusion += f"群体 {cluster}：{count} 人 ({percentage:.1f}%)\n"

        conclusion += "\n【三、各群体特征分析】\n"
        conclusion += "-" * 40 + "\n"

        # 分析每个聚类的特征
        for cluster in sorted(set(self.cluster_result)):
            cluster_data = self.data[self.data['聚类标签'] == cluster]

            conclusion += f"\n▶ 客户群体 {cluster}：\n"

            # 计算关键指标
            if '消费频率' in cluster_data.columns:
                avg_freq = cluster_data['消费频率'].mean()
                conclusion += f"  • 平均消费频率：{avg_freq:.2f} 次\n"

            if '总消费金额' in cluster_data.columns:
                avg_monetary = cluster_data['总消费金额'].mean()
                conclusion += f"  • 平均消费金额：¥{avg_monetary:,.2f}\n"

            if '会员年限' in cluster_data.columns:
                avg_years = cluster_data['会员年限'].mean()
                conclusion += f"  • 平均会员年限：{avg_years:.1f} 年\n"

            if '满意度评分' in cluster_data.columns:
                avg_satisfaction = cluster_data['满意度评分'].mean()
                conclusion += f"  • 平均满意度：{avg_satisfaction:.2f}/5.0\n"

            # 客户群体标签
            if '消费频率' in cluster_data.columns and '总消费金额' in cluster_data.columns:
                avg_freq = cluster_data['消费频率'].mean()
                avg_monetary = cluster_data['总消费金额'].mean()

                overall_freq_mean = self.data['消费频率'].mean()
                overall_monetary_mean = self.data['总消费金额'].mean()

                if avg_freq > overall_freq_mean and avg_monetary > overall_monetary_mean:
                    label = "高价值客户"
                    strategy = "重点维护，提供VIP服务，增加客户粘性"
                elif avg_freq > overall_freq_mean and avg_monetary <= overall_monetary_mean:
                    label = "活跃客户"
                    strategy = "提升客单价，推荐高价值产品"
                elif avg_freq <= overall_freq_mean and avg_monetary > overall_monetary_mean:
                    label = "潜力客户"
                    strategy = "增加消费频次，推送优惠活动"
                else:
                    label = "一般客户"
                    strategy = "通过营销活动激活，提升消费意愿"

                conclusion += f"  • 客户类型：{label}\n"
                conclusion += f"  • 建议策略：{strategy}\n"

        # 营销建议
        conclusion += "\n【四、营销策略建议】\n"
        conclusion += "-" * 40 + "\n"
        conclusion += """
1. 高价值客户维护策略：
   - 建立专属客户经理服务
   - 提供优先登机、贵宾厅等增值服务
   - 定期推送个性化优惠方案

2. 潜力客户提升策略：
   - 设计积分翻倍活动，刺激消费频次
   - 推送中高端产品推荐
   - 开展会员日专属优惠

3. 活跃客户转化策略：
   - 推出套餐组合优惠
   - 交叉销售高毛利产品
   - 建立消费积分兑换体系

4. 一般客户激活策略：
   - 发送新人优惠券
   - 开展限时促销活动
   - 优化产品性价比
"""

        conclusion += "\n" + "=" * 60 + "\n"
        conclusion += f"报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        self.conclusion_text.insert(1.0, conclusion)

    def generate_report(self):
        """生成分析报告"""
        if self.data is None or self.cluster_result is None:
            messagebox.showwarning("警告", "请先完成聚类分析")
            return

        # 切换到结论标签页
        self.notebook.select(2)
        messagebox.showinfo("提示", "分析报告已生成，请查看「分析结论」标签页")

    def export_report(self):
        """导出报告"""
        if not self.conclusion_text.get(1.0, tk.END).strip():
            messagebox.showwarning("警告", "请先生成分析报告")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.conclusion_text.get(1.0, tk.END))
                messagebox.showinfo("成功", f"报告已保存至：{file_path}")
            except Exception as e:
                messagebox.showerror("失败", f"保存报告时出错：{str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = CustomerAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
