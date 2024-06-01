import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

from SVM import generate_data, train_svm, plot_decision_boundary

# 在终端输入 streamlit run ./main.py 启动
st.title("支持向量机实验可视化")

# 初始化或获取session_state中的状态
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'dataX' not in st.session_state:
    st.session_state.dataX = None
if 'dataY' not in st.session_state:
    st.session_state.dataY = None


with st.sidebar:
    st.markdown("## 数据设置")
    n_samples = st.number_input("训练数据量", 10, 10000, 100, 10)
    position = st.number_input("两组数据中心距离调节", 1, 10, 2, 1)
    step = st.select_slider("步长", [0.00001, 0.0001, 0.001, 0.01, 0.1])
    C = st.number_input("C值", 0.1, 100.0, 0.1, 0.1)
    num_iters = st.number_input("迭代次数", 10, 100000, 10000, 10000)

    if st.button("生成数据"):
        center1 = np.zeros(2)
        center2 = np.zeros(2)
        for i in range(2):
            if np.random.random() < 0.5:
                center1[i] = position
            else:
                center2[i] = position

        dataX, dataY = generate_data(center1, center2, n_samples)
        # 保存生成的数据和状态
        st.session_state.dataX = dataX
        st.session_state.dataY = dataY
        st.session_state.data_generated = True

if st.session_state.data_generated:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(st.session_state.dataX[st.session_state.dataY.flatten() == 1][:, 0],
               st.session_state.dataX[st.session_state.dataY.flatten() == 1][:, 1],
               color='g', marker='o', label='class 1')
    ax.scatter(st.session_state.dataX[st.session_state.dataY.flatten() == -1][:, 0],
               st.session_state.dataX[st.session_state.dataY.flatten() == -1][:, 1],
               color='b', marker='*', label='class 2')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('DataSet')
    ax.legend()
    st.pyplot(fig)

    if st.button("开始训练"):
        w, b, support_vectors = train_svm(st.session_state.dataX, st.session_state.dataY,
                                          learning_rate=step, C=C, num_iters=num_iters)
        st.pyplot(plot_decision_boundary(st.session_state.dataX, st.session_state.dataY,
                                         w, b, support_vectors))


st.write("感谢streamlit提供的可视化工具！")
