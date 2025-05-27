from typing import Optional
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack


class ArxivPaper:
    def __init__(self, paper: arxiv.Result):
        self._paper = paper
        self.score = None

    @property
    def title(self) -> str:
        return self._paper.title

    @property
    def summary(self) -> str:
        return self._paper.summary

    @property
    def authors(self) -> list[str]:
        return self._paper.authors

    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())

    @property
    def pdf_url(self) -> str:
        return self._paper.pdf_url

    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(
                f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count', 0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(
                f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count', 0) == 0:
            return None
        return repo_list['results'][0]['url']

    @cached_property
    def tex(self) -> dict[str, str]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            file = self._paper.download_source(dirpath=tmpdirname)
            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                return None

            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None

            bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
            match len(bbl_file):
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl', '')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                        main_tex = None
                case _:
                    logger.debug(
                        f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                    main_tex = None
            if main_tex is None:
                logger.debug(
                    f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
            # read all tex files
            file_contents = {}
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode('utf-8', errors='ignore')
                # remove comments
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(
                    r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '',
                                 content, flags=re.DOTALL)
                # remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                content = re.sub(r'\\\\', '', content)
                # remove consecutive spaces
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(
                        f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content

            if main_tex is not None:
                main_source: str = file_contents[main_tex]
                # find and replace all included sub-files
                include_files = re.findall(
                    r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                for f in include_files:
                    if not f.endswith('.tex'):
                        file_name = f + '.tex'
                    else:
                        file_name = f
                    main_source = main_source.replace(
                        f'\\input{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
            else:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                file_contents["all"] = None
        return file_contents

    @cached_property
    def tldr(self) -> str:
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            # remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            # remove figure
            content = re.sub(
                r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            # remove table
            content = re.sub(
                r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            # find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(
                r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(
                r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        prompt = """根据一篇论文的标题、摘要、引言以及结论（如果有的话），生成一句纯文本的 __LANG__ TLDR 总结。请确保只返回总结内容，不要包含任何 LaTeX 或其他格式化命令。
        
        论文标题: __TITLE__
        摘要: __ABSTRACT__
        引言: __INTRODUCTION__
        结论: __CONCLUSION__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        tldr = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "你是一位科研助手，能够完美地总结论文内容，并向用户提供论文的核心思想。你的总结必须是所指定语言的单个句子。总结必须是纯文本。请勿在回应中使用任何 LaTeX、Markdown、HTML 或任何其他标记语言。请勿包含 \\documentclass、\\usepackage、\\begin{tcolorbox}、\\textbf 或类似的格式命令。",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return tldr

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            # search for affiliations
            possible_regions = [r'\\author.*?\\maketitle',
                                r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL)
                       for p in possible_regions]
            match = next((m for m in matches if m), None)
            if match:
                information_region = match.group(0)
            else:
                logger.debug(
                    f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None
            prompt = f"请根据以下 LaTeX 格式的论文作者信息，提取作者的单位，并以 Python列表的格式返回，列表应按作者顺序排序。如果未找到单位信息，请返回空列表 '[]'。作者信息如下：\n{information_region}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
            prompt = enc.decode(prompt_tokens)
            llm = get_llm()
            affiliations = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位能够完美从论文作者信息中提取作者单位的助手。你应该返回一个按作者顺序列出的 Python 单位列表，例如 ['TsingHua University','Peking University']。如果一个单位包含多层级信息，例如 'Department of Computer Science, TsingHua University'，你应仅返回顶级单位 'TsingHua University'。不要包含重复的单位。如果没有找到单位信息，则返回空列表 []。你只需要返回最终的单位列表，不要返回任何中间结果。",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            try:
                affiliations = re.search(
                    r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
                affiliations = eval(affiliations)
                affiliations = list(set(affiliations))
                affiliations = [str(a) for a in affiliations]
            except Exception as e:
                logger.debug(
                    f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affiliations
