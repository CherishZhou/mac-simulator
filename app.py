import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 网页全局配置 =================
st.set_page_config(page_title="MAC 全工况性能评估实验室", layout="wide")
st.title("🎛️ MAC 算法全工况性能评估实验室")
st.markdown("✅ **模型**：完全复刻 600步、6大突变工况、带外部扰动的系统。并引入 **ITAE+能量惩罚** 综合代价 $J$ 作为客观量化评价标准。")

# 设置 Matplotlib 中文字体 (防止方块乱码)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 侧边栏：4D 参数控制台 =================
st.sidebar.header("🛠️ 结构参数")
N_val = st.sidebar.slider("截断长度 N", min_value=20, max_value=100, value=82, step=1)
P_val = st.sidebar.slider("预测时域 P", min_value=5, max_value=40, value=32, step=1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 性能参数")
alpha_val = st.sidebar.slider("柔化因子 α", min_value=0.100, max_value=0.990, value=0.796, step=0.001, format="%.3f")
lam_val = st.sidebar.slider("控制权重 λ", min_value=0.010, max_value=2.000, value=0.110, step=0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.success("💡 综合评价指标 J 越小，代表系统在所有工况下的综合表现越完美。")

# ================= 3. 核心 600 步考场级仿真引擎 =================
@st.cache_data # 使用缓存加速页面滑动响应
def run_evaluation_simulation(N, P, alpha, lam):
    Ts = 1.0
    MAX_STEPS = 600
    
    # 1. 标称模型生成 (控制器内部模型永远是标称的)
    s_nom = np.zeros(N)
    for k in range(N):
        t = k * Ts
        if t >= 3.0: s_nom[k] = -1.0 * (1 - np.exp(-(t - 3.0) / 10.0))

    # 2. 构建控制矩阵
    S_mat = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1):
            if (i - j) < N: S_mat[i, j] = s_nom[i - j]
            else: S_mat[i, j] = s_nom[N - 1]

    try:
        H_inv = np.linalg.inv(S_mat.T @ S_mat + lam * np.eye(P)) @ S_mat.T
        h_vec = H_inv[0, :]
    except np.linalg.LinAlgError:
        return None, None, None, None, 999999.0, True

    # 数组初始化
    y = np.zeros(MAX_STEPS)
    u = np.zeros(MAX_STEPS)
    w_sim = np.zeros(MAX_STEPS)
    delta_u_hist = np.zeros(MAX_STEPS)
    sp_sim = np.zeros(MAX_STEPS)
    
    for t in range(MAX_STEPS):
        sp_sim[t] = 1.0 if (t // 100) % 2 == 0 else -1.0

    # 真实工况跳变逻辑
    def get_true_params(t_step):
        if t_step < 100:   return -1.0, 10.0, 3.0  
        elif t_step < 200: return -2.0, 10.0, 3.0  
        elif t_step < 300: return -2.0, 20.0, 3.0  
        elif t_step < 400: return -2.0, 20.0, 6.0  # 危险滞后区
        elif t_step < 500: return -0.5, 5.0, 1.0   
        else:              return -1.0, 10.0, 3.0  

    J_cost = 0.0
    is_diverged = False

    # 3. 闭环循环
    for t in range(1, MAX_STEPS - 1):
        K_true, T_true, tau_true = get_true_params(t)
        d_true = int(tau_true / Ts)
        a1 = np.exp(-Ts / T_true)
        b1 = K_true * (1 - a1)
        
        u_delayed = u[t - 1 - d_true] if (t - 1 - d_true) >= 0 else 0.0
        y[t] = a1 * y[t-1] + b1 * u_delayed
        
        # 外部扰动
        if 50 <= t < 100: y[t] += 0.5

        # 自由响应预测
        Y_free = np.zeros(P)
        for i in range(P):
            free_resp = y[t]
            for j in range(1, N):
                idx_1 = min(i + j, N - 1)
                idx_2 = min(j, N - 1)
                free_resp += (s_nom[idx_1] - s_nom[idx_2]) * delta_u_hist[t - j]
            Y_free[i] = free_resp

        # 参考轨迹柔化
        W = np.zeros(P)
        w_temp = y[t]
        for i in range(P):
            w_temp = alpha * w_temp + (1 - alpha) * sp_sim[t]
            W[i] = w_temp
        w_sim[t] = W[0]

        # 求解控制增量
        delta_U_opt = h_vec @ (W - Y_free)
        du_clip = np.clip(delta_U_opt, -5.0, 5.0)
        u_next = np.clip(u[t-1] + du_clip, -20.0, 20.0)
        
        u[t] = u_next
        delta_u_hist[t] = u_next - u[t-1]
        
        # ====== 核心：实时代价评估 ======
        error = abs(sp_sim[t] - y[t])
        if error > 8.0:
            J_cost = 999999.0
            is_diverged = True
            break
            
        t_rel = t % 100
        J_cost += (t_rel * error) + 10.0 * abs(delta_u_hist[t])
        
    y[-1], w_sim[-1], u[-1] = y[-2], w_sim[-2], u[-2]
    return y, u, w_sim, sp_sim, J_cost, is_diverged

# 执行仿真计算
y_res, u_res, w_res, sp_res, current_J, diverged = run_evaluation_simulation(N_val, P_val, alpha_val, lam_val)

# ================= 4. UI 渲染：KPI 计分板 =================
st.markdown("### 🏆 核心性能评价")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if diverged:
        st.error("🚨 系统已发散 (DIVERGED)！代价 J 爆表！")
        st.metric(label="全工况综合评价代价 J", value="> 999,999")
    elif y_res is None:
        st.error("🚨 控制矩阵奇异，无法计算！")
    else:
        # 如果接近咱们的最优解 5142，给予绿色高亮提示
        if current_J < 5500:
            st.success(f"🌟 极致性能！当前参数表现优异。")
        st.metric(label="全工况综合评价代价 J (越小越好)", value=f"{current_J:,.2f}")

st.markdown("---")

# ================= 5. 动态绘图渲染 =================
if y_res is not None and not diverged:
    fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
    t_axis = np.arange(600)

    # --- 上图：系统输出 ---
    ax_y.plot(t_axis, sp_res, 'k--', label='设定值 SP')
    ax_y.plot(t_axis, w_res, 'g:', alpha=0.8, linewidth=2, label='柔化轨迹 W')
    ax_y.plot(t_axis, y_res, 'r-', linewidth=2, label='系统真实输出 Y')
    
    # 绘制分界线与危险高亮
    for border in range(100, 600, 100):
        ax_y.axvline(x=border, color='gray', linestyle=':', alpha=0.5)
    ax_y.axvspan(300, 400, color='red', alpha=0.1, label='危险工况: τ=6 极端滞后')
    
    ax_y.set_title(f'600步全工况输出响应 (当前参数: N={N_val}, P={P_val}, α={alpha_val:.3f}, λ={lam_val:.3f})', fontsize=14)
    ax_y.set_ylabel('输出幅值 Y', fontsize=12)
    ax_y.set_xlim(0, 600)
    ax_y.set_ylim(-3.5, 3.5)
    ax_y.legend(loc='upper right')
    ax_y.grid(True, linestyle='--', alpha=0.5)

    # --- 下图：控制器输出 ---
    ax_u.step(t_axis, u_res, 'b-', where='post', label='控制量 U')
    for border in range(100, 600, 100):
        ax_u.axvline(x=border, color='gray', linestyle=':', alpha=0.5)
    ax_u.axvspan(300, 400, color='red', alpha=0.1)
    
    ax_u.set_title('控制器动作', fontsize=12)
    ax_u.set_xlabel('时间步 (k)', fontsize=12)
    ax_u.set_ylabel('控制量 U', fontsize=12)
    ax_u.set_xlim(0, 600)
    ax_u.set_ylim(-15, 15)
    ax_u.legend(loc='upper right')
    ax_u.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
elif diverged and y_res is not None:
    # 即使发散了，也画出前几百步看看是怎么死的
    fig, ax_y = plt.subplots(figsize=(14, 4))
    t_axis = np.arange(600)
    ax_y.plot(t_axis, y_res, 'r-', label='发散轨迹')
    ax_y.set_title('系统发散前轨迹')
    ax_y.set_ylim(-10, 10)
    st.pyplot(fig)
