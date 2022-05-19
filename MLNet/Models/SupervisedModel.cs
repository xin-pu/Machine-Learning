using AutoDiff;
using Numpy;
using YAXLib.Attributes;

namespace MLNet.Models
{
    /// <summary>
    ///     无监督学习模型
    /// </summary>
    public abstract class SupervisedModel : Model
    {
        #region Core

        /// <summary>
        ///     参数
        /// </summary>
        [YAXDontSerialize]
        public Variable[] Variables { protected set; get; } = { };

        #endregion

        /// <summary>
        ///     小批量随机梯度下降
        /// </summary>
        /// <param name="traindatas_x"></param>
        /// <param name="traindatas_y"></param>
        /// <param name="trainPlan"></param>
        public override void Fit(
            NDarray traindatas_x,
            NDarray traindatas_y,
            TrainPlan trainPlan)
        {
            try
            {
                printTitle($"{Name} Start Fit");
                print($"{trainPlan}");

                /// Step 1 Convert Model
                traindatas_x = Transform.Call(traindatas_x);

                /// Step 2 Initial Variables and Temp Weights
                InitialWeights(traindatas_x, traindatas_y);

                /// Step 3 Fit
                foreach (var epoch in Enumerable.Range(0, trainPlan.Epoch))
                {
                    var final_batch = trainPlan.Batch == 0 ? traindatas_x.len : trainPlan.Batch;
                    var steps = (int) Math.Ceiling(1.0 * traindatas_x.shape[0] / final_batch);

                    foreach (var batch in Enumerable.Range(0, steps))
                    {
                        var index = $"{batch * final_batch}:{(batch + 1) * final_batch},:";
                        var batch_x = traindatas_x[index];
                        var batch_y = traindatas_y[index];

                        /// get grad and loss at this step
                        var (grad_batch, _) = CostFunc.Call(Resolve, batch_x, batch_y);

                        /// update weight by optimizer
                        Resolve = Optimizer.Call(Resolve, grad_batch, epoch);
                    }

                    var (_, epochloss) = CostFunc.Call(Resolve, traindatas_x, traindatas_y);

                    var pred_y = call(traindatas_x, traindatas_y.shape);
                    Evaluate(traindatas_y, pred_y);

                    if (Print)
                    {
                        var metrics = string.Join(' ', Metrics.Select(a => a.ToString()));
                        Log.print?.Invoke($"Epoch:\t{epoch:D4}\tVal Loss:{epochloss:F4}\t{metrics}\t{Resolve}");
                    }
                }
            }
            catch (Exception ex)
            {
                print($"{Name} meet Exception:\r\n{ex.Message}");
            }
        }

        public override void InitialWeights(NDarray traindatas_x, NDarray trandatas_y)
        {
            Variables = initialVariables(traindatas_x, trandatas_y);
            // Todo Add different function for weights 
            Resolve = np.random.random_(Variables.Length);
            CostFunc.GiveVariables(Variables);
        }

        #region internal

        internal virtual Variable[] initialVariables(NDarray x, NDarray y)
        {
            var featureCount = x.shape[1];
            var variables = Enumerable.Range(0, featureCount).Select(_ => new Variable()).ToArray();
            return variables;
        }

        #endregion
    }
}