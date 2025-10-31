import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { HttpModule } from '@nestjs/axios';
import { OllamaController } from './ollama.controller';
import { OllamaService } from './ollama.service';

@Module({
  imports: [ConfigModule, HttpModule],
  controllers: [OllamaController],
  providers: [OllamaService]
})
export class OllamaModule {}
