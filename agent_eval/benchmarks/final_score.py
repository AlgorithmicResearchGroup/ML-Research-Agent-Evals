class FinalScore:
    def __init__(self, results, num_tokens_used, time_in_seconds, task, model_averages):
        self.num_tokens_used = num_tokens_used
        self.time_in_seconds = time_in_seconds
        self.results = results
        self.task = task
        self.model_averages = model_averages
        
        
    def get_normalized_tokens_used(self):
        normalized_value = self.num_tokens_used/self.model_averages['tokens_used']
        return normalized_value
    
    def get_normalized_time_in_seconds(self):
        normalized_value = self.time_in_seconds/self.model_averages['time_in_seconds']
        return normalized_value
        
    def get_normalized_tokens_per_second(self):
        for metric, value in self.results:
            if metric == "tokens_per_second":
                normalized_value = value/self.model_averages['tokens_per_second']
                return normalized_value
            
            
    def get_normalized_perplexity(self):
        for metric, value in self.results:
            if metric == "perplexity":
                normalized_value = value/self.model_averages['perplexity']
                return normalized_value
            
    def get_normalized_latency(self):
        for metric, value in self.results:
            if metric == "latency":
                normalized_value = value/self.model_averages['latency']
                return normalized_value
            
    def get_normalized_rouge_score(self):
        for metric, value in self.results:
            if metric == "rouge_l":
                normalized_value = value/self.model_averages['rouge_l']
                return normalized_value
            
            
    def type_0_score(self):
        """
        normalied_score = 1 / ((P / P_avg) * (Tu / Tu_avg) * (Tc / Tc_avg))
        """
        print("Using Type 0 Score")
        norm_perplexity = self.get_normalized_perplexity()
        norm_time_in_seconds = self.get_normalized_time_in_seconds()
        norm_num_tokens = self.get_normalized_tokens_used()
        print(f"Norm Perplexity {norm_perplexity}")
        print(f"Num Tokens {self.num_tokens}")
        print(f"Time in Seconds {self.time_in_seconds}")
        final_score  = 1 / (norm_perplexity * norm_num_tokens * norm_time_in_seconds)
        return final_score
    
    def type_1_score(self):
        """
        normalied_score = (Ts / Ts_avg) / ((P / P_avg) * (Tu / Tu_avg) * (Tc / Tc_avg))
        """
        print("Using Type 1 Score")
        norm_perplexity = self.get_normalized_perplexity()
        norm_tokens_per_second = self.get_normalized_tokens_per_second()
        norm_num_tokens = self.get_normalized_tokens_used()
        norm_time_in_seconds = self.get_normalized_time_in_seconds()
        
        final_score  = norm_tokens_per_second / (norm_perplexity * norm_num_tokens * norm_time_in_seconds)
        return final_score
    
    def type_2_score(self):
        """
        normalied_score = (Ts / Ts_avg) / ((P / P_avg) * (Tu / Tu_avg) * (Tc / Tc_avg) * (R / R_avg))
        """
        print("Using Type 2 Score")
        norm_perplexity = self.get_normalized_perplexity()
        norm_tokens_per_second = self.get_normalized_tokens_per_second()
        norm_latency = self.get_normalized_latency()
        norm_rouge_score = self.get_normalized_rouge_score()
        final_score  = norm_tokens_per_second / (norm_perplexity * norm_latency * norm_rouge_score * self.num_tokens * self.time_in_seconds)
        return final_score
                
     
    def calculate_final_score(self):
        if "data_augmentation" in self.task:
            self.final_result = self.type_0_score()
        else:
            self.final_result = self.type_1_score()
        return self.final_result
        
    def print_markdown_table(self, final_result):
        header = "| Metric                      | Value       |\n"
        separator = "|-----------------------------|-------------|\n"
        rows = "\n".join([f"| {metric:<27} | {value:<11} |" for metric, value in self.results])
        final_row = f"| {'Final Result':<27} | {str(final_result):<11} |"
        table = f"{header}{separator}{rows}\n{final_row}"
        print(table)