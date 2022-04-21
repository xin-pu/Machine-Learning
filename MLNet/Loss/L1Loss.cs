﻿using AutoDiff;
using MLNet.Utils;
using Numpy;

namespace MLNet.Loss
{
    /// <summary>
    ///     J(la)= sigma(|y-yp|)
    /// </summary>
    public class L1Loss : LossBase
    {
        public L1Loss(string name, Variable[] variables, NDarray x, NDarray y)
            : base(name, variables, x, y)
        {
        }

        public int Features { set; get; }

        internal override Term createLoss(Variable[] variables, NDarray x, NDarray y)
        {
            var batchsize = x.shape[0];
            Features = x.shape[1];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var yp = y[$"{i},:"].GetData<double>();
                var xp = np2.matmul(rowX, variables);
                var delta = xp - yp[0];
                var abs = TermBuilder.Power(TermBuilder.Power(delta, 2), 0.5);
                return abs;
            });
            return TermBuilder.Sum(list);
        }
    }
}