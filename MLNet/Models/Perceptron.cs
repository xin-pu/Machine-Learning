using AutoDiff;
using Numpy;
using Numpy.Models;
using YAXLib.Attributes;

namespace MLNet.Models
{
    public class Perceptron : Model
    {
        public Perceptron(int[] classes)
        {
            Classes = classes;
        }


        internal Dictionary<int, Variable[]> initialVariables(NDarray x, NDarray y)
        {
            //确定分类数量
            var features = x.shape[1];
            var variablesDict = Enumerable.Range(0, Classes.Length)
                .ToDictionary(i => i, i => Enumerable.Range(0, features).Select(_ => new Variable()).ToArray());

            return variablesDict;
        }

        internal void InitialWeights(NDarray traindatas_x, NDarray trandatas_y)
        {
            var features = traindatas_x.shape[1];
            Variables = Enumerable.Range(0, Classes.Length)
                .ToDictionary(i => i, i => Enumerable.Range(0, features).Select(_ => new Variable()).ToArray());

            // Todo Add different function for weights 
            Resolve = np.random.random_(Classes.Length, Variables.Count);
            CostFunc.GiveVariables(Variables.SelectMany(a => a.Value).ToArray());
        }


        public override void Fit(NDarray traindatas_x, NDarray traindatas_y, TrainPlan trainPlan)
        {
            try
            {
                printTitle($"{Name} Start Fit");
                print($"{trainPlan}");

                /// Step 1 Convert Model
                traindatas_x = Transform.Call(traindatas_x);


                /// Step 2 Initial Variables and Temp Weights
                InitialWeights(traindatas_x, traindatas_y);
            }
            catch (Exception ex)
            {
                print($"{Name} meet Exception:\r\n{ex.Message}");
            }
        }

        internal override NDarray call(NDarray x, Shape shape)
        {
            throw new NotImplementedException();
        }

        #region Core

        /// <summary>
        ///     分类标签
        /// </summary>
        [YAXDontSerialize]
        public int[] Classes { internal set; get; } = null!;

        /// <summary>
        ///     参数
        /// </summary>
        [YAXDontSerialize]
        public Dictionary<int, Variable[]> Variables { internal set; get; } = null!;

        #endregion
    }
}