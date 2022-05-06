namespace MLNet.Models.Regression
{
    public enum Constraint
    {
        None = 0,
        L1 = 1,
        L2 = 2,
        LP = 3,
        Ridge = 2,
        Lasso = 1,
        ElasticNet = 3
    }
}