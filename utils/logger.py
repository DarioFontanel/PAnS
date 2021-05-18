import logging
import os


class Logger:

    def __init__(self, logdir, rank, type='torch', filename=None, summary=True, step=None):
        self.logger = None
        self.type = type
        self.rank = rank
        self.step = step
        self.logdir_results = os.path.join("logs", "results")
        self.summary = summary
        if summary:
            if type == 'tensorboardX':
                import tensorboardX
                self.logger = tensorboardX.SummaryWriter(logdir)
            elif type == "torch":
                from torch.utils.tensorboard import SummaryWriter
                self.logger = SummaryWriter(logdir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'


        logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:rank{rank}: %(message)s')

        if rank == 0:
            os.makedirs(self.logdir_results, exist_ok=True)
            logging.info(f"[!] starting logging at directory {logdir}")


    def close(self):
        if self.logger is not None:
            self.logger.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_figure(tag, image, step)

    def add_table(self, tag, tbl, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.logger.add_text(tag, tbl_str, step)

    def add_results(self, results, tag="Results"):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            text = "<table width=\"100%\">"
            for k, res in results.items():
                text += f"<tr><td>{k}</td>" + " ".join([str(f'<td>{x}</td>') for x in res.values()]) + "</tr>"
            text += "</table>"
            self.logger.add_text(tag, text)

    def print(self, msg):
        logging.info(msg)

    def info(self, msg):
        if self.rank == 0:
            logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    def log_results(self, task, name, results, novel=False):
        if self.rank == 0:
            file_name = f"{task}.csv" if not novel else f"{task}_novel.csv"
            path = f"{self.logdir_results}/{file_name}"
            text = [name]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            with open(path, "a") as file:
                file.write(row)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def is_not_none(self):
        return self.type != "None"

