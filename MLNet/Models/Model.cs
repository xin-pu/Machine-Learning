using System.Text;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Optimizers;
using MLNet.Transforms;
using Numpy;
using Numpy.Models;
using YAXLib;
using YAXLib.Attributes;

namespace MLNet.Models
{
    /// <summary>
    ///     Learning Model
    /// </summary>
    public abstract class Model : ViewModelBase
    {
        protected Model()
        {
            Name = GetType().Name;
            Transform = new LinearFirstorder();
        }


        public string Name { protected set; get; }
        public string FilePath => $"{Name}.xml";
        public bool Print { protected set; get; } = true;


        #region MyRegion

        /// <summary>
        ///     模型变换
        /// </summary>
        [YAXDontSerialize]
        public Transform Transform { protected set; get; }

        /// <summary>
        ///     解
        /// </summary>
        [YAXDontSerialize]
        public NDarray Resolve { set; get; } = null!;

        /// <summary>
        ///     优化器
        /// </summary>
        [YAXDontSerialize]
        public Optimizer Optimizer { protected set; get; } = null!;

        /// <summary>
        ///     损失函数
        /// </summary>
        [YAXDontSerialize]
        public Loss CostFunc { protected set; get; } = null!;

        /// <summary>
        ///     评估器
        /// </summary>
        [YAXDontSerialize]
        public Metric[] Metrics { protected set; get; } = { };

        /// <summary>
        ///     赋优化器
        /// </summary>
        /// <param name="optimizer"></param>
        public virtual void GiveOptimizer(Optimizer optimizer)
        {
            Optimizer = optimizer;
        }

        /// <summary>
        ///     赋模型变换
        /// </summary>
        /// <param name="transform"></param>
        public virtual void GiveTransform(Transform transform)
        {
            Transform = transform;
        }

        /// <summary>
        ///     赋损失函数
        /// </summary>
        /// <param name="loss"></param>
        public virtual void GiveLoss(Loss loss)
        {
            CostFunc = loss;
        }

        /// <summary>
        ///     赋评估器
        /// </summary>
        /// <param name="metrics"></param>
        public virtual void GiveMetric(params Metric[] metrics)
        {
            Metrics = metrics;
        }

        #endregion

        #region Public Function for Model

        public abstract void Fit(
            NDarray traindatas_x,
            NDarray traindatas_y,
            TrainPlan trainPlan);


        public abstract void InitialWeights(NDarray traindatas_x, NDarray trandatas_y);

        /// <summary>
        ///     评估
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public void Evaluate(NDarray y_true, NDarray y_pred)
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
            var x_transformed = Transform.Call(x);

            /// Step 2 Predict 
            var y_pred = call(x_transformed, new Shape(x.shape[0], 0));

            return y_pred;
        }


        internal abstract NDarray call(NDarray x, Shape shape);

        #endregion


        #region print

        internal void print(object obj)
        {
            if (Print)
                Log.print?.Invoke(obj);
        }


        internal void printTitle(string title)
        {
            Log.print?.Invoke($"{new string('-', 10)}{title}{new string('-', 10)}");
        }

        public override string ToString()
        {
            var strBuild = new StringBuilder();
            strBuild.AppendLine($"{Name}");
            strBuild.AppendLine($"Resolve:\r\n{Resolve}");
            return strBuild.ToString();
        }

        #endregion


        #region serialize

        public virtual void Save(string path)
        {
            using var stream = File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var textWriter = new StreamWriter(stream);
            var serializer = new YAXSerializer(typeof(Model));
            serializer.Serialize(this, textWriter);
            Resolve.tofile(FilePath, "", "");
            textWriter.Flush();
        }

        public static Model Load(string path)
        {
            using var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var textReader = new StreamReader(stream);
            var type = typeof(Model);
            var deserializer = new YAXSerializer(type);
            var model = (Model) deserializer.Deserialize(textReader);
            var r = np.fromfile(model.FilePath);
            model.Resolve = np.expand_dims(r, -1);
            return model;
        }

        #endregion
    }
}