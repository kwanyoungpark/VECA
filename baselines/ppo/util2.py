import tensorflow as tf
class ParamBackup:
    def __init__(self, sess, src_scopes, dst_scopes, backup_subsets = []):
        assert len(src_scopes) == len(dst_scopes)
        self.copy_op, self.revert_op = [], []
        self.sess = sess
        self.backup_subsets = backup_subsets
        for src, dst in zip(src_scopes, dst_scopes):
            main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=src)
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst)
            self.copy_op.append([target_var.assign(main_var.value()) for main_var, target_var in zip(main_vars, target_vars)])
            self.revert_op.append([main_var.assign(target_var.value()) for main_var, target_var in zip(main_vars, target_vars)])

    def commit(self):
        self.sess.run(self.copy_op)
        for backup in self.backup_subsets:
            backup.commit()

    def revert(self):
        self.sess.run(self.revert_op)
        for backup in self.backup_subsets:
            backup.revert()