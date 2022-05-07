using AutoDiff;
using MLNet.Kernels;
using MLNet.LearningModel;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Optimizers;
using MLNet.Utils;
using Numpy;
using YAXLib.Attributes;

namespace MLNet.Models
{
    public abstract class SupervisedModel : Model
    {
        /// <summary>
        ///     小批量随机梯度下降
        /// </summary>
        /// <param name="traindatas_x"></param>
        /// <param name="traindatas_y"></param>
        /// <param name="trainConfig"></param>
        public void Fit(
            NDarray traindatas_x,
            NDarray traindatas_y,
            TrainConfig trainConfig)
        {
            try
            {
                printTitle($"{Name} Start Fit");
                print($"{trainConfig}");

                /// Step 1 Convert Model
                traindatas_x = transform(traindatas_x);

                /// Step 2 Initial Variables and Temp Weights
                InitialWeights(traindatas_x, traindatas_y);

                /// Step 3 Fit
                foreach (var epoch in Enumerable.Range(0, trainConfig.Epoch))
                {
                    var final_batch = trainConfig.Batch == 0 ? traindatas_x.len : trainConfig.Batch;
                    var steps = (int) Math.Ceiling(1.0 * traindatas_x.shape[0] / final_batch);

                    foreach (var batch in Enumerable.Range(0, steps))
                    {
                        var index = $"{batch * final_batch}:{(batch + 1) * final_batch},:";
                        var batch_x = traindatas_x[index];
                        var batch_y = traindatas_y[index];

                        /// get grad and loss at this step
                        var (grad_batch, loss) = CostFunc.Call(Resolve, batch_x, batch_y);

                        /// update weight by optimizer
                        Resolve = Optimizer.Call(Resolve, grad_batch);
                    }

                    var (_, epochloss) = CostFunc.Call(Resolve, traindatas_x, traindatas_y);

                    Evaluate(traindatas_y, call(traindatas_x));

                    if (Print)
                    {
                        var metrics = string.Join(' ', Metrics.Select(a => a.ToString()));
                        Log.print?.Invoke($"Epoch:\t{epoch:D4}\tVal Loss:{epochloss:F4}\t{metrics}");
                    }
                }
            }
            catch (Exception ex)
            {
                print($"{Name} Predict:\r\n{ex.Message}");
            }
        }

        internal void InitialWeights(NDarray traindatas_x, NDarray trandatas_y)
        {
            Variables = initialVariables(traindatas_x, trandatas_y);
            Resolve = np.random.randn(Variables.Length);
            CostFunc.GiveVariables(Variables);
        }

        #region Core

        /// <summary>
        ///     模型变换
        /// </summary>
        [YAXDontSerialize]
        public Kernel Kernel { set; get; } = null!;

        /// <summary>
        ///     损失函数
        /// </summary>
        [YAXDontSerialize]
        public Loss CostFunc { protected set; get; } = null!;

        /// <summary>
        ///     优化器
        /// </summary>
        [YAXDontSerialize]
        public Optimizer Optimizer { protected set; get; } = null!;

        /// <summary>
        ///     参数
        /// </summary>
        [YAXDontSerialize]
        public Variable[] Variables { protected set; get; } = { };

        /// <summary>
        ///     赋优化器
        /// </summary>
        /// <param name="optimizer"></param>
        public void GiveOptimizer(Optimizer optimizer)
        {
            Optimizer = optimizer;
        }

        /// <summary>
        ///     赋损失函数
        /// </summary>
        /// <param name="loss"></param>
        public void GiveLoss(Loss loss)
        {
            CostFunc = loss;
        }

        /// <summary>
        ///     赋评估器
        /// </summary>
        /// <param name="metrics"></param>
        public void GiveMetric(params Metric[] metrics)
        {
            Metrics = metrics;
        }

        /// <summary>
        ///     评估
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        internal void Evaluate(NDarray y_true, NDarray y_pred)
        {
            if (Metrics.Length == 0) return;

            Metrics.ToList().ForEach(m => { m.Call(y_true, y_pred); });
        }

        /// <summary>
        ///     预测
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public NDarray Predict(NDarray x)
        {
            /// Step 1 Transform
            var x_cvt = transform(x);

            /// Step 2 Casll 
            var y_pred = call(x_cvt);

            return y_pred;
        }

        #endregion

        #region internal

        internal virtual NDarray transform(NDarray x)
        {
            var kernel = Kernel.Transform(x);
            var final = transformer.to_linear_firstorder(kernel);
            return final;
        }

        internal abstract Variable[] initialVariables(NDarray x, NDarray y);

        internal abstract NDarray call(NDarray x);

        #endregion
    }
}