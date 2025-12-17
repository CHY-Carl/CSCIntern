def write_results_to_excel(sheet, bs_res, mc_res, fdm_res, is_series=False):
    """
    统一负责将三种引擎的计算结果写入 Excel。
    """
    # 1. 提取价格
    prices = [
        [float(bs_res['price'])],
        [float(mc_res['price'])],
        [float(fdm_res['price'])]
    ]
    if is_series:
        # Series 模式下，price 是向量，需要转置或堆叠
        # 假设输入已经是向量，这里稍作处理
        # 实际上 series 模式下我们会在外部组装好大矩阵，这里主要服务 Single
        pass
    else:
        # Single 模式写入 B7:D7 (转置写入)
        # 注意：Excel 是一行行写的，所以我们要构造 [[bs, mc, fdm]] 或者分开写
        # 之前的布局是 B7=BS, C7=MC, D7=FDM (Call), B8... (Put)
        # 这里为了通用性，我们假设外部处理好 Call/Put 的分别调用
        pass

    # 这里为了不破坏你现有的 Excel 布局逻辑，我们直接在 run 函数里写写入逻辑更直观
    # 这个辅助函数暂时略过，直接看下面的核心实现。